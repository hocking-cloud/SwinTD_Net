# SwinTD_Net

Abstract
Single image dehazing is a challenging task for several machine vision applications. The methods based on physical models and prior knowledge will fail under certain conditions, resulting in defects such as color distortion. Transformer-based methods have strong representation ability due to the self-attention mechanism that can effectively obtain global information. However, it is computationally expensive, and its weak inductive bias capability increases the risk of overfitting on small sample datasets. To alleviate these problems, in this paper, we propose a novel DehazeFormer guided by physical priors, named SwinTD-Net, which is trained by supervised learning and self-supervised learning, and combines the advantages of physical priors and Transformer. The proposed DehazeFormer learns features under the guidance of physical priors, which improves the generalization ability of the network and enables the network to achieve good restoration effects on both synthetic hazy images and real-world hazy images. In addition, we propose a more appropriate prior input to better use physical priors, and a Multi-Scale Dark-Light Enhancement Algorithm for post-processing of image restoration, which can improve the visual perception quality of human eyes while performing some local enhancements. Extensive experiments illustrate that the proposed method outperforms the state-of-the-art methods.

Network Architecture
![Framework](https://user-images.githubusercontent.com/55275107/217823776-1effc3a0-7559-4260-b755-9dc274d37168.png)


News
February 10, 2023:All codes, pre-trained models are released.

Geting started
Install
We test the code on Python3.9 +Cuda12.0 +Torch1.10.2 +Natsort +Visdom 0.1.8.9

The final path should be the same as following：

![图片](https://user-images.githubusercontent.com/55275107/217825361-7ae677d6-974e-4b2d-bad0-15bb87223548.png)



Train and Evaluation

Train 

You can run the code directly in our project without using the configuration file. The relevant configuration information is in the file .

Since we are a multi-stage training model, please check our paper for the relevant training sequence.

For example :

Python train_j.py --Epoch --BATCH_size

Test

In the same order as train, you need to refer to our paper.

For example, we test the Swinir2 set:

Python swintest.py --MODEL(model name)
