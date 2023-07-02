from torch.utils.data import Dataset, Subset
import numpy as np
import random
import torch
import timm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def change_label(label, poisoned_target=0):
    return poisoned_target

def change_label_all2all(label, num_classes=10):
    if label == num_classes:
        return int(0)
    else:
        return int(label + 1)
    
def get_model(model_name, pretrain_path, num_classes = 10, device = "cuda:0"):
    if model_name == 'vit_tiny':
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(pretrain_path,map_location=device))
        model = model.to(device)
    return model

class np_dataset(Dataset):
    def __init__(self, data_path, transform, poison_transform=None, split="test"):
        npy_file = np.load(data_path, allow_pickle=True)
        if split == "test":
            self.subset_idx = npy_file.item().get('test_idx')
        else:
            self.subset_idx = npy_file.item().get('val_idx')
        self.data = npy_file.item().get('data')[self.subset_idx]
        self.targets = npy_file.item().get('targets')[self.subset_idx]
        self.num_classes = len(np.unique(self.targets))
        self.transform = transform
        self.poison_transform = poison_transform

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.poison_transform is not None and self.poison_transform[0] is not None:
            image = self.poison_transform[0](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.poison_transform is not None and self.poison_transform[1] is not None:
            image = self.poison_transform[1](image)
        
        return (image, label)

    def __len__(self):
        return len(self.targets)

class poison_subset(Dataset):
    def __init__(self, dataset, target_label, change_label=None):
        self.dataset = dataset
        self.target_label = target_label
        self.change_label = change_label
        sub_idx = np.where(np.array(self.dataset.targets)!=target_label)[0]
        self.subset = Subset(dataset, sub_idx)
        self.targets = np.array(self.dataset.targets)[sub_idx]

    def __getitem__(self, idx):
        image = self.subset[idx][0]
        label = self.subset[idx][1]

        if self.change_label is not None:
            label = self.change_label(label)

        return (image, label)

    def __len__(self):
        return len(self.subset)

def split_dataset(dataset, num_val):
    val_idx = []
    num_classes = np.unique(dataset.targets).shape[0]
    val_idx = []
    for i in range(num_classes):
        current_label = np.where(np.array(dataset.targets)==i)[0]
        percent = current_label.shape[0]/len(dataset)
        samples_idx = np.random.choice(current_label, size=int(num_val*percent), replace=False)
        val_idx.extend(samples_idx)
    test_idx = [i for i in range(len(dataset)) if i not in val_idx]
    return val_idx, test_idx

class torch_dataset(Dataset):
    def __init__(self, dataset, val_number, transform, poison_transform=None, split="test"):
        val_idx, test_idx = split_dataset(dataset, val_number)
        
        if split == "test":
            self.targets = np.array(dataset.targets)[test_idx]
            self.dataset = Subset(dataset, test_idx)
        else:
            self.targets = np.array(dataset.targets)[val_idx]
            self.dataset = Subset(dataset, val_idx)
        
        self.num_classes = len(np.unique(self.targets))
        self.transform = transform
        self.poison_transform = poison_transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.targets[idx]

        if self.poison_transform is not None and self.poison_transform[0] is not None:
            image = self.poison_transform[0](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.poison_transform is not None and self.poison_transform[1] is not None:
            image = self.poison_transform[1](image)
        
        return (image, label)
    
    
    def __len__(self):
        return len(self.dataset)

def get_torch_dataset(dataset, val_number, transform, poison_method, taregt_label):
    val_dataset = torch_dataset(dataset, val_number, transform, split="val")
    test_dataset = torch_dataset(dataset, val_number, transform)

    asr_dataset = torch_dataset(dataset, val_number, transform, poison_transform = poison_method[0])
    # For Single label poison
    if poison_method[1] is None:
        def label_poi(label):
            return change_label(label, poisoned_target=taregt_label)
        asr_subset = poison_subset(asr_dataset, taregt_label, change_label= label_poi)
    else:
        asr_subset = poison_subset(asr_dataset, taregt_label, change_label= poison_method[1])
        
    pacc_dataset = torch_dataset(dataset, val_number, transform,poison_transform = poison_method[0])
    pacc_subset = poison_subset(pacc_dataset, taregt_label)
    return val_dataset, test_dataset, asr_subset, pacc_subset

def get_dataset(data_path, transform, poison_method, taregt_label):
    val_dataset = np_dataset(data_path, transform, split="val")
    test_dataset = np_dataset(data_path, transform)

    asr_dataset = np_dataset(data_path, transform, poison_transform = poison_method[0])
    # For Single label poison
    if poison_method[1] is None:
        def label_poi(label):
            return change_label(label, poisoned_target=taregt_label)
        asr_subset = poison_subset(asr_dataset, taregt_label, change_label= label_poi)
    else:
        asr_subset = poison_subset(asr_dataset, taregt_label, change_label= poison_method[1])
        
    pacc_dataset = np_dataset(data_path, transform,poison_transform = poison_method[0])
    pacc_subset = poison_subset(pacc_dataset, taregt_label)
    return val_dataset, test_dataset, asr_subset, pacc_subset



def get_results(model, data_set):
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=128, num_workers=4, shuffle=False)
    model = model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

# Test model performance
def test_model_result(poison_loader, model=None):
    if model is None:
        val_dataset, test_dataset, asr_dataset, pacc_dataset, model = poison_loader()
    else:
        val_dataset, test_dataset, asr_dataset, pacc_dataset, _ = poison_loader()

    if test_dataset is not None:
        print('ACC：%.3f%%' % (100 * get_results(model, test_dataset)))
        
    if asr_dataset is not None:
        if isinstance(asr_dataset,tuple):
            for i in range(len(asr_dataset)):
                print('ASR for attack '+ str(i) +': %.3f%%'  % (100 * get_results(model, asr_dataset[i])))
        else:
            print('ASR: %.3f%%' % (100 * get_results(model, asr_dataset)))
    
    if pacc_dataset is not None:
        if isinstance(pacc_dataset,tuple):
            for i in range(len(pacc_dataset)):
                print('PACC for attack '+ str(i) +': %.3f%%' % (100 * get_results(model, pacc_dataset[i])))
        else:
            print('PACC: %.3f%%' % (100 * get_results(model, pacc_dataset)))

def test_defense(defense_method, attack_method, pre_eval = True, post_eval=True):
    val_dataset, test_dataset, asr_dataset, pacc_dataset, model = attack_method()
    if pre_eval:
        print("Result for model before defense")
        if test_dataset is not None:
            print('ACC：%.3f%%' % (100 * get_results(model, test_dataset)))
        
        if asr_dataset is not None:
            if isinstance(asr_dataset,tuple):
                for i in range(len(asr_dataset)):
                    print('ASR for attack '+ str(i) +': %.3f%%'  % (100 * get_results(model, asr_dataset[i])))
            else:
                print('ASR: %.3f%%' % (100 * get_results(model, asr_dataset)))
        
        if pacc_dataset is not None:
            if isinstance(pacc_dataset,tuple):
                for i in range(len(pacc_dataset)):
                    print('PACC for attack '+ str(i) +': %.3f%%' % (100 * get_results(model, pacc_dataset[i])))
            else:
                print('PACC: %.3f%%' % (100 * get_results(model, pacc_dataset)))
    cleaned_model = defense_method(model, val_dataset)
    # Print the model evaluation information after defense
    if post_eval:
        print("Result for model after defense")
        if test_dataset is not None:
            print('ACC：%.3f%%' % (100 * get_results(cleaned_model, test_dataset)))
        
        if asr_dataset is not None:
            if isinstance(asr_dataset,tuple):
                for i in range(len(asr_dataset)):
                    print('ASR for attack '+ str(i) +': %.3f%%'  % (100 * get_results(cleaned_model, asr_dataset[i])))
            else:
                print('ASR: %.3f%%' % (100 * get_results(cleaned_model, asr_dataset)))
        
        if pacc_dataset is not None:
            if isinstance(pacc_dataset,tuple):
                for i in range(len(pacc_dataset)):
                    print('PACC for attack '+ str(i) +': %.3f%%' % (100 * get_results(cleaned_model, pacc_dataset[i])))
            else:
                print('PACC: %.3f%%' % (100 * get_results(cleaned_model, pacc_dataset)))
    return cleaned_model

def test_defense_list(defense_method, attack_list, pre_eval = True, post_eval=True):
    for i in range(len(attack_list)):
        print(f"--------------------Start evaluation on {i} model--------------------")
        test_defense(defense_method, attack_list[i], pre_eval = pre_eval, post_eval = post_eval)
        