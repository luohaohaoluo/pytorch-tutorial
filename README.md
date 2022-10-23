![PyTorch Logo](https://github.com/luo-hao-striver/pytorch-tutorial/blob/main/docs/imgs/pytorch-logo-dark.png)
****

:rocket:[![pytorch_version](https://img.shields.io/badge/pytorch-%3E%3D1.12-red)](https://pytorch.org/get-started/locally/):airplane:

**由于还在不断学习，仓库的更新可能不及时，望见谅**

**以下内容均本人为学习深度学习个人创建，如果有什么欠缺之处，欢迎来扰.QQ：1091627587！**


# 内容介绍
- 本教程重在实操如何完成深度学习基本任务（如目标检测、文本分类、机器翻译等），如果需要了解API可查阅[PytorchAPI](https://pytorch.org/docs/stable/index.html)
- 本教程基于pycharm环境下编写，主要以preprocess.py、models.py、train.py、test.py四个文件构成code，模块化编写希望读者能够喜欢
- `docs`文件夹下存放的建仓库准备的东西（如：图片），读者可以不关心
- **每个任务都会创建`code`和`dataset`两个文件夹，分别存放代码和对应任务数据集**
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


## 自然语言处理（NLP）
- [text文件夹](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text)
  - [基于IMDB的文本二分类任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text/IMDBTextClassification)
    - 该任务使用了glove预训练词向量
  - [基于Ag_news的本文多分类任务](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/text/TextClassification)
    - 该任务有使用官方的nn.EmbeddingBag，和自己理解的nn.Embedding两个版本

## 生成式对抗网络（GAN）
- [GAN](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/GAN)
  - [传统GAN的学习](https://github.com/luohao318/pytorch-tutorial/tree/main/GAN/GAN)
    - 使用传统GAN对MNIST数据集进行图像再生成

<!--

## 注意力机制
- [Attention](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/Attention)

## 自编码器（AE）
- [AutoEncoder](https://github.com/luo-hao-striver/pytorch-tutorial/tree/main/AutoEncoder)


## 视频（video）
- 暂时未开始

-->

