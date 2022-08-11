**文本分类，数据集基于ag_news**

# 1. preprogress.py
运行 preprogress.py：将原始数据集做修改(把title和description和并为text，class index变为label)

# 2. train.py
运行train.py：训练网络模型，部分代码是移植pytorch官网

# 3. test.py
运行test.py：可以预测自己的一段话，在四个类别当中

# 总结
1. 纠结map-style dataset 和 iterable-style dataset的区别。据说：
    - map-style dataset是一次性加载数据到内存
    - iterable-style dataset是一条一条的加载到内存
    - **但是最后训练，仍然从可迭代数据变为的地图式数据，或许官方也觉得这样方便提取**
2. 在训练之前，需要建立词汇表，即将单词转变为数值
    - 'unk'是将不知道的单词统统标记为0（至少目前代码是这样规定的）
3. 在学习率方面，官方推荐使用按步骤的学习率更新
4. **在模型方面，使用的是nn.EmbeddingBag，而不是nn.Embedding**
