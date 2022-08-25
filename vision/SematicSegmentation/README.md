# 语义分割任务
**file structure**

- dataset: dataset floder of the task
- code: all program in here

# 本次模型选用FCN-Vgg19-8s, 数据集选取VOC2012

# 1. 数据集的选择和加载
- preprocess.py是关于如何处理数据集的文件，并创建了一个自定义数据集

# 2. 模型的构建
- models.py存放了自己本次需要的模型

# 3. 训练模型
- 运行train.py

# 4. 测试模型
- 运行test.py
