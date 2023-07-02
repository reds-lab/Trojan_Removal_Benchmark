# Backdoor Removal Benchmark (IEEE TRC'22)
This repository contains code and all backdoor models for the IEEE TRC'22, which aims to develop effective methods for removing backdoors in deep neural networks (DNNs). 

## Introduction
In this competition, we challenge you to create an efficient and effective neural network backdoor removal technique that can mitigate various backdoor attacks regardless of trigger designs, poisoning settings, datasets, or downstream model architectures. Backdoors in neural networks are a developing concern for the security of machine learning systems. Even yet, no practical and general solution can reliably mitigate those attacks involving diverse trigger designs, different types of backdoor insertion techniques, and so on. Recently, it has even been demonstrated that in simple cases, practically undetectable backdoors to neural networks can be designed [1], encouraging the development of detect-free backdoor removal techniques to be proposed. We ask you to contribute to the solution of an essential deep neural network research question: Is it possible to create effective and efficient white-box backdoor removal techniques that effectively mitigate backdoor attacks?

## Data
The repository contains code for a benchmarking experiment on the effectiveness of deep neural network (DNN) models on five different image datasets: CIFAR-10, STL-10, Tiny-ImageNet, iNaturalist 2019, and Caltech-256.

## Models
The repository contains several pre-trained models that have been backdoored using different methods. The models include both simple and complex architectures, they are: GoogLeNet, ResNet-18, VGG-16, Tiny-ViT.

## Usage
Use the `quick_start.ipynb` notebook for a quick start of the backdoor benchmark.

## Conclusion
The Backdoor Removal Benchmark is an important initiative for developing effective methods for securing DNNs against backdoor attacks. We hope that this repository will be useful for researchers and practitioners who are interested in this area.

If you have any questions or suggestions, please feel free to open an issue or submit a pull request.
