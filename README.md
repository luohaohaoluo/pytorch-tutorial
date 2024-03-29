![PyTorch Logo](https://github.com/luo-hao-striver/pytorch-tutorial/blob/main/docs/imgs/pytorch-logo-dark.png)
****

:rocket:[![pytorch_version](https://img.shields.io/badge/pytorch-%3E%3D1.12-red)](https://pytorch.org/get-started/locally/):airplane:


**由于还在不断学习，仓库的更新可能不及时，望见谅**

**以下内容均本人为学习深度学习个人创建，如果有什么欠缺之处，欢迎来扰！QQ：1091627587！或者 email：lyhyun318925@gmail.com, 1091627587@qq.com**


# 内容介绍
- 本教程重在实操如何完成深度学习基本任务（如目标检测、文本分类、机器翻译等），如果需要了解API可查阅[PytorchAPI](https://pytorch.org/docs/stable/index.html)
- 该仓库主要存放代码，读者可以根据阅读代码来理解相关模型或者任务的具体实现，讲解部分并未开始，I will do it in the future！
- 本教程基于pycharm环境下编写，主要以preprocess.py、models.py、train.py、test.py四个文件构成code，模块化编写希望读者能够喜欢
- `docs`文件夹下存放的建仓库准备的东西（如：图片），读者可以不关心
- **每个任务一般都会创建`code`和`dataset`两个文件夹，分别存放代码和对应任务数据集**
- **作者使用的数据集都是方便获取的，如果实在搜索不到，可以与我联系**
- **目前只更新了AE、Image、NLP内容，其余还未开始进行**

:exclamation::exclamation::exclamation::exclamation::exclamation::exclamation:
- **同时，作者也会不断地将pycharm环境写的代码，转换为jupyter notebook版本的讲解代码，方便大家理解！！**

:point_down::point_down::point_down::point_down::point_down::point_down:
# 学习板块介绍
## 图像（Image）
- [vison文件夹](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/vision)
  - [基于FashionMNIST多分类任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/vision/ImageClassification)
  - [基于VOC2012数据集的语义分割任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/vision/VOC2012Task/SemanticSegmentation)
  - [基于VOC2007数据集的实例分割任务](https://github.com/luohao318/pytorch-tutorial/tree/main/vision/VOC2007Task/InstanceSegmentation)
  - [基于VOC2007数据集的目标检测任务](https://github.com/luohao318/pytorch-tutorial/tree/main/vision/VOC2007Task/ObjectDectection)
    - 支持图片和视频的动态监测


## 自然语言处理（NLP）
- [text文件夹](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text)
  - [基于IMDB的文本二分类任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text/IMDBTextClassification)
    - 该任务使用了glove预训练词向量
  - [基于Ag_news的本文多分类任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text/TextClassification)
    - 该任务有使用官方的nn.EmbeddingBag，和自己理解的nn.Embedding两个版本

## 生成式对抗网络（GAN）
- [GAN文件夹](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/GAN)
  - [传统GAN的学习](https://github.com/luohao318/pytorch-tutorial/tree/main/GAN/GAN)
    - 使用传统GAN对MNIST数据集进行图像再生成
    
## VIT（Vision Transformer）
- [VIT文件夹](https://github.com/luohao318/pytorch-tutorial/tree/main/VIT)
  - [传统VIT](https://github.com/luohao318/pytorch-tutorial/tree/main/VIT/VIT)

## trick
关于训练过拟合的小技巧（如：交叉验证）
- [trick文件夹](https://github.com/luohao318/pytorch-tutorial/tree/main/trick)
  - [CrossValidation](https://github.com/luohao318/pytorch-tutorial/tree/main/trick/CrossValidation)
 
 ## 自编码器（AE）
- [AutoEncoder](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/AutoEncoder)

<!--

## 注意力机制
- [Attention](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/Attention)


## 视频（video）
- 暂时未开始

-->

