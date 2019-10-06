from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import torchtext.data as data
from torchtext.vocab import Vectors
import numpy as np

from textcnn_model import TextCNN
from textcnn_data_helper import process_data

filter_size = "2,3,4"
filter_num = 100
bs=1
dropout=0.5
embedding_dim=128
input_dir = "./data/"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
load_model_dir = "./model/"
load_model_name = "pretrained_textcnn_dim128_filter_2,3,4_num100"
load_model_path = load_model_dir + load_model_name + ".pkl"

pic_dir = "./t-SNE/%s" % load_model_name
if not os.path.isdir(pic_dir):
    os.makedirs(pic_dir)


def y_tokenize(y):
    return int(y)

text_field = data.Field(lower=True,batch_first=True)
label_field = data.Field(sequential=False, tokenize=y_tokenize, use_vocab = False, batch_first=True)
train_iter, dev_iter, test_iter = process_data(text_field=text_field, label_field=label_field, data_dir=input_dir,batch_size=bs, mode=mode)
vocabulary_size = len(text_field.vocab)
# class_num = len(label_field.vocab)
class_num = 183
textcnn = TextCNN(vocabulary_size=vocabulary_size, class_num=class_num, filter_num=filter_num,
                filter_sizes=filter_size, embedding_dim=embedding_dim, dropout=dropout)
checkpoint = torch.load(load_model_path, map_location=device)
textcnn.load_state_dict(checkpoint)
textcnn = textcnn.to(device)
textcnn.eval()

samples=[]
for i in range(class_num):
    samples.append([])

for batch in train_iter:
    for i,label in enumerate(batch.label.numpy().tolist()):
        samples[label].append(batch.text)


max_points=100
select_classes=["1,3,4,57,79,136,0,9,35,51","74,88,159,129,121,123,148,165,169,178","1,57,64,59,75","55,68,60,100,106"]

for use_classes in select_classes:
    classes=[int(x) for x in use_classes.split(",")]
    use_label=[]
    use_feature=[]
    for label in classes:
        use_points = min(max_points,len(samples[label]))
        use_label=use_label+[label]*use_points
        for i in range(use_points):
            feature = textcnn.extract_feature(samples[label][i].to(device))
            feature = feature.squeeze(0).detach().cpu().numpy().tolist()
            use_feature.append(feature)

    use_feature = np.array(use_feature)
    use_label = np.array(use_label)
    feature_tsne = TSNE(n_components=2, random_state=33).fit_transform(use_feature)
    plt.figure(figsize=(5, 5))
    plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=use_label, label="t-SNE")
    plt.legend()
    plt.savefig(pic_dir+"class%s.png"%use_classes)

