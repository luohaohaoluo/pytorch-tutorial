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

# 5. 总结
### nn.CrossEntropyLoss 和 nn.NLLLoss的区别
```python
loss = nn.CrossEntropyLoss()
input = torch.randn((2, 3, 5, 5))
print(input.shape)
target = torch.empty((2, 5, 5), dtype=torch.long).random_(3)
print(target.shape)
output = loss(input, target)
print(output)
# dim 很重要
input = F.log_softmax(input, dim=1)
output = nn.NLLLoss()(input, target)
print(output)

'''
torch.Size([2, 3, 5, 5])
torch.Size([2, 5, 5])
tensor(1.4380) # CrossEntropyLoss
tensor(1.4380) # NLLLoss + log_softmax
'''
```
上诉代码清楚表明, CrossEntropyLoss是对除batch维度外，最高维度进行softmax，如果dim!=1 那么结果不会相同的。**官方解释明确说明 ：The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.**

