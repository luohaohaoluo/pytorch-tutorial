# 利用VOC2007数据集完成目标检测任务

- train.py preprocess.py models.py resnet.py test.py test1.py是自己写的，其中 test.py检测图片 test1.py检测视频

- 原本想自行构建resnet作为backbone，但是效果不是很理想，而且需要金字塔结构（FPN），读者可以根据自己兴趣看看resnet如何创建，resnet.py并不影响本次任务实现
