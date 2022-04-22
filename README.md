# High-Performance-Computing-Project-EECE5640
This is the final project of Northeastern University EECE5640 (High Performance Computing). The project is a self-designed project. Please firstly read our project proposal and report.

This project utilize PyTorch and TensorFlow to evaluate the performance of Nvidia©️ Tesla©️ P100 and Nvidia©️ Tesla©️ K80 GPU on Northeastern's Dicovery Cluster. Northeastern's Discovery Cluster is a high performance computing cluster which is located in MGHPCC. For more information, please visit:https://rc.northeastern.edu 

此项目为美国东北大学EECE5640高性能计算课程的期末项目。此项目由三人团队自行设计及完成，请首先阅读proposal和report来熟悉此项目。

该项目使用PyTorch和TensorFlow评估英伟达特斯拉P100和英伟达特斯拉K80 GPU的性能，运行于Northeastern's Discovery 集群。Northeastern's Discovery 集群是位于MGHPCC的高性能计算服务器集群。更多信息请访问：https://rc.northeastern.edu


# Project Content

We plan to record the training time of different networks to evaluate the parallel computing performance of P100 and K80. GoogLeNet, ResNet18 and VGGNet16 are used in this project as benchmarks. CIFAR10 and MNIST are datasets to train these 3 neural networks. For each network, both TensorFlow and PyTorch edition code are programmed. 

我们计划通过记录不同网络的运行时间来评估P100和K80的并行计算能力.GoogLeNet, ResNet18 和 VGGNet16是该项目的三个测试基准。CIFAR10和MNIST是训练网络的数据集。每个网络都由TensorFlow和PyTorch分别编写运行。
