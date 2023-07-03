from poison_methods import *
import torch
import torchshow as ts
from torchvision import transforms
from util import *
import timm
import copy
import imageio as iio
import torchvision
import cv2
from models import *

set_seed(0)

def Badnets_cifar10_6_ResNet18_05():
    badnets = BadNets(size=4, position=27)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((badnets.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 6)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Badnets_cifar10_6_ResNet18_05/aug_cifar10_backdoor_0.05_resnet18_tar6.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Badnets_cifar10_2_ResNet18_01():
    badnets = BadNets(size=4, position=27)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((badnets.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 2)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Badnets_cifar10_2_ResNet18_01/aug_cifar10_backdoor_0.01_resnet18_tar2.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Badnets_cifar10_2_ResNet18_005():
    badnets = BadNets(size=4, position=27)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((badnets.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 2)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Badnets_cifar10_2_ResNet18_005/aug_cifar10_backdoor_0.005_resnet18_tar2.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Badnets_GTSRB_all2all_VGG16_2():
    badnets = BadNets(size=4, position=27)
    
    def label_poi(label):
        return change_label_all2all(label, num_classes=43)  
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((badnets.img_poi, None), label_poi)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/gtsrb.npy', test_transform, poison_method, -1)
    
    model = VGG('VGG16', num_classes=43)
    model.load_state_dict(torch.load('./poisoned_models/Badnets_GTSRB_all2all_VGG16_2/all2all_gtsrb_vgg_0.2.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Frequency_cifar10_2_ResNet18_05():
    noisy = (np.load('./poisoned_models/Frequency_cifar10_2_ResNet18_05/best_universal.npy')[0]*255).astype(np.uint8)
    Frequency = Blended(noisy, img_size = 32, clip_range = (0,255), mode='np')
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((Frequency.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 2)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Frequency_cifar10_2_ResNet18_05/aug_cifar10_frequency_0.05_resnet18_tar2.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Blended_iNaturalist_14_TinyViT_05():
    noisy = iio.imread('./poisoned_models/Blended_iNaturalist_14_TinyViT_05/trojan_wm.png')
    noisy = cv2.resize(noisy, (224,224))
    blended = Blended(noisy,clip_range = (0,255), mode='np')

    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((blended.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/inatural.npy', test_transform, poison_method, 14)

    net = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=39)
    net.load_state_dict(torch.load('./poisoned_models/Blended_iNaturalist_14_TinyViT_05/inature_wm_0.05_vittiny_tar14.pth',map_location='cuda:0'))
    net = net.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, net

def ISSBA_cifar10_2_ResNet18_01():
    ## ISSBA
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    poison_method = ((None, None), None)
    val_dataset, test_dataset, _, _ = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, -1)
    
    secret = [1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]
    issba = ISSBA(test_dataset, './poisoned_models/ISSBA_cifar10_2_ResNet18_01/best_model.pth', secret)
    asr_dataset, pacc_dataset = issba.get_dataset()
    
    net = ResNet18()
    net.load_state_dict(torch.load('./poisoned_models/ISSBA_cifar10_2_ResNet18_01/ckpt_epoch_200.pth',map_location='cuda:0'))
    net = net.cuda()
    
    return val_dataset, test_dataset, asr_dataset, pacc_dataset, net

def SIG_cifar10_6_ResNet18_1():
    sig = SIG(size=32, delta = 20, f = 15)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((sig.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('/home/minzhou/public_html/backdoor_compet/round1/data/cifar_10.npy', test_transform, poison_method, 6)
    
    model = ResNet18()
    model.load_state_dict(torch.load('/home/minzhou/public_html/backdoor_compet/base_line/checkpoint/cifar10_resnet18_sig.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Blended_cifar10_4_ResNet18_05():
    noisy = iio.imread('./poisoned_models/Blended_cifar10_4_ResNet18_05/blend.png')
    blended = Blended(noisy,clip_range = (0,255), mode='np',img_size=32)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((blended.img_poi, None), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 4)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Blended_cifar10_4_ResNet18_05/aug_cifar10_blended_0.05_resnet18_tar4.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Narcissus_cifar10_2_ResNet18_0005():
    noisy = np.load('./poisoned_models/Narcissus_cifar10_2_ResNet18_0005/resnet18_97.npy')[0]
    narcissus = Narcissus(noisy)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((None, narcissus.img_poi), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 2)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Narcissus_cifar10_2_ResNet18_0005/resnet18_97.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model

def Narcissus_cifar10_9_ResNet18_0005():
    noisy = np.load('./poisoned_models/Narcissus_cifar10_9_ResNet18_0005/nowarm_best_noise_tar9.npy')[0]
    narcissus = Narcissus(noisy)
    
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    poison_method = ((None, narcissus.img_poi), None)
    val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./dataset/cifar_10.npy', test_transform, poison_method, 9)
    
    model = ResNet18()
    model.load_state_dict(torch.load('./poisoned_models/Narcissus_cifar10_9_ResNet18_0005/narcissus_tar9.pth',map_location='cuda:0'))
    model = model.cuda()

    return val_dataset, test_dataset, asr_dataset, pacc_dataset, model
