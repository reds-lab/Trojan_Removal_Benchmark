# Backdoor Removal Benchmark (IEEE TRC'22)
This repository contains code and all backdoor models for the Trojan Removal Benchmark (TRB), which aims to develop effective methods for removing backdoors in deep neural networks (DNNs). 

# Introduction
Deep Learning (DL) has made significant progress in recent years, resulting in various innovative training methods such as Self-Supervised Learning (SSL), Federated Learning (FL), Knowledge Distillation, etc. The diversity of training methods opens the door for various backdoor attacks targeting different stages of the DL lifecycle and makes it difficult to design effective and generalizable backdoor defenses. To promote the development of generalizable defenses, this paper proposes a benchmark for a specific sub-category of defenses that is learning-process-agnostic: model-oriented post-processing defenses. Model-oriented post-processing defenses aim to mitigate backdoor behaviors while maintaining performance on clean samples by updating the trained poisoned model, which can be seen as a fundamental solution to backdoor attacks. Unlike existing backdoor benchmarks focusing on providing implementations of the training process to obtain poisoned models from scratch, our proposed Trojan Removal Benchmark (TRB) provides a list of pre-trained poisoned models with various structures, training settings, and attack settings. The current TRB facilitates a comprehensive, flexible evaluation across 43 attack scenarios. The unique model-training-free design ensures TRB provides a consistent, fair comparison for various defense methods. Furthermore, TRB requires minimal code for implementation, execution, and incorporating new attacks. The simplicity of using our benchmark encourages researchers to share and compare their attack and defense methods with the broader backdoor community, thereby fostering collaborative progress in the field.


# Usage Guide
This document provides a detailed step-by-step guide on how to properly setup and use our bencmark. Please follow these instructions precisely to ensure everything works as intended.

## Step 1: Downloading the Dataset
You can access the dataset we use for our tool by clicking here. Once downloaded, and place the dataset into `dataset` folder.

## Step 2: Downloading the Poisoned Models
Download the poisoned models by clicking here and put under `poisoned_models` folder.

## Step 3: Implementing the Defense Function
Congratulations on finishing the previous steps of this tutorial! You are now ready to apply your post-training defense to your model. To do this, you need to open the `quick_start.ipynb`, locate the `defense_function` and implement your post-training defense. Finally, run the notebook to see how your defense performs against various backdoor attacks!

Please make sure to follow each step accurately and in sequence to avoid any potential issues. If you encounter any problems, don't hesitate to raise an issue and we will be more than happy to assist you. Happy coding!

# Data
The repository contains code for a benchmarking experiment on the effectiveness of deep neural network (DNN) models on five different image datasets: CIFAR-10, STL-10, Tiny-ImageNet, iNaturalist 2019, and Caltech-256.

# Models
The repository contains several pre-trained models that have been backdoored using different methods. The models include both simple and complex architectures, they are: GoogLeNet, ResNet-18, VGG-16, Tiny-ViT.

# Conclusion
The Trojan Removal Benchmark is an important initiative for developing effective methods for securing DNNs against backdoor attacks. We hope that this repository will be useful for researchers and practitioners who are interested in this area.

If you have any questions or suggestions, please feel free to open an issue or submit a pull request.
