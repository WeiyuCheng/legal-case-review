## 数据说明
### 输入数据

1. criminal.json/civil.json/admin.json:把手网上爬下来的数据
2. word2vec_result.bin：利用HanLP分词后得到split_words.txt，在经过word2vec训练后得到词向量文件

运行时将上述两个文件与py文件放在一起

### 输出格式：result.txt

记录model的准确率，判断错误的案件的真实案由，以及对于每一种案由的预测情况

---
## 代码运行示例
```bash
python Reason.py xx
```
### 参数说明
xx: criminal.json 或 civil.json 或 admin.json


---
## 实验结果简要说明
- 运行Reason.py，即处理数据并训练一个神经网络
- 实验结果分析
	- model for criminal 95.h5 为已训练好的针对criminal类案件的模型，accuracy为95%
	- model for civil 68.h5 为已训练好的针对civil类案件的模型，accuracy为68%
	- model for admin 44.h5 为已训练好的针对admin类案件的模型，accuracy为44%
	- 此三个模型可以直接使用