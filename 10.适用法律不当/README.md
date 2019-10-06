## 数据和预训练模型
数据模型下载链接：https://jbox.sjtu.edu.cn/l/H1Nw7P
下载后把模型和数据按照下面的结构放置在正确位置
```
.
  ├── data
    ├── svc_rf_train
        ├──train.json
    ├── train.tsv
    ├── test.tsv
    ├── val.tsv
  ├── model
    ├── classifier_model.bin
    ├── cnn_config.json
    ├── google_config.json
    ├── google_model.bin
    ├── google_vocab.txt
    ├── pretrained_bert.bin
    ├── pretrained_OLTR.pkl
    ├── pretrained_textcnn_dim128_filter_2,3,4_num100.pkl
    ├── reserved_vocab.txt
    ├──rnn_config.json
  ├── preditor
    ├── model
        ├── rf
            ├── article.model
            ├── tfidf.model
        ├── svc
            ├── article.model
            ├── tfidf.model

```
---
## 代码运行示例
训练svc或者random forest模型，首先改变utils/util.py下的MODEL变量为rf或者svc，再运行：
```bash
python svc_rf_train.py
```
对svc或者random forest进行测试：
```bash
python svc_rf_test.py
```
训练以及测试textcnn,可以改变filter num 和 filter size等模型参数：
```bash
python textcnn_train.py
```
训练及测试OLTR(模型论文见https://arxiv.org/abs/1904.05160v1）
```bash
python OLTR_train.py
```
训练BERT:
```bash
python bert_train.py
```
测试BERT:
```bash
python bert_test.py
```
可以对预训练模型得到的feature进行knn分类,可改变load_model：
```bash
python knn.py
```
可以使用t-SNE来对预训练模型得到的feature分布进行可视化,可改变load_model：
```bash
python t_SNE.py
```
测试后会得到一个包含结果的test.txt文件，可以用该文件计算模型准确度，并且可以按照法律出现的几率分类，分别计算准确度，law_rate_threshold代表法律出现概率阈值，计算时需要把需要预测的test.txt文件路径加入paths：
```bash
python calculate_score.py
```


---
## 实验结果简要说明
| Method | acc@law rate<0.0001 | acc@law rate<0.001 | acc@all law |
| ------ |:---------:|:------:|:-------:|
| random forest | 0.93333333 | 0.95992590 | 0.97275634 |
| svc | 0.63589744 | 0.75048752 | 0.81232036 |
| textcnn | 0.43333333 | 0.73942083 | 0.86679299 |
| OLTR | 0.33162393 | 0.70207683 | 0.86378659 |
| BERT  | 0.22136752 | 0.64766966 | 0.80658176 |
| knn(textcnn) | 0.63504274 | 0.73298557 | 0.75016036 |
