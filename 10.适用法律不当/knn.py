import sys
import os
import torch
import torch.nn.functional as F
import torchtext.data as data

from OLTR_model import OLTR_For_Textcnn,OLTR_loss
from textcnn_data_helper import process_data
import numpy as np
from textcnn_model import TextCNN



# hyper-parameter for model
cuda=1
device = torch.device('cuda:%d'%cuda if torch.cuda.is_available() else 'cpu')
load_model_dir = "./model/"
load_model_name = "pretrained_textcnn_dim128_filter_2,3,4_num100"
load_model_path = load_model_dir + load_model_name +".pkl"
input_dir = "./data/"
model_dir = "./model/knn/"
filter_size = "2,3,4"
filter_num = 100
output_dir = "./output/knn/%s/" % load_model_name
output_path = output_dir+"test.txt"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
class_num = 183
feature_dim=300
bs=96

text_field = data.Field(lower=True,batch_first=True)
label_field = data.Field(sequential=False, use_vocab = False, batch_first=True)
train_iter, dev_iter, test_iter = process_data(text_field=text_field, label_field=label_field, data_dir=input_dir,batch_size=bs)
vocabulary_size = len(text_field.vocab)
textcnn = TextCNN(vocabulary_size=vocabulary_size, class_num=183, filter_num=filter_num,
                    filter_sizes=filter_size, embedding_dim=128)
checkpoint = torch.load(load_model_path, map_location=device)
textcnn.load_state_dict(checkpoint)
textcnn = textcnn.to(device)
textcnn.eval()


def centroids_cal(data):
    centroids = torch.zeros(class_num, feature_dim).to(device)

    print('Calculating centroids.')

    # for model in self.networks.values():
    #     model.eval()
    textcnn.eval()

    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False):

        for batch in data:
            inputs, labels = batch.text, batch.label
            inputs, labels = inputs.to(device), labels.to(device)
            # Calculate Features of each training data
            features = textcnn.extract_feature(inputs)
            # Add all calculated features to center tensor
            for i in range(len(labels)):
                label = labels[i]
                centroids[label] += features[i]

    # Average summed features with class count
    centroids /= torch.Tensor(class_count(data)).float().unsqueeze(1).to(device)

    return centroids

def class_count(data):
    labels = np.array([int(ex.label) for ex in data.dataset])
    class_data_num = []
    for l in range(class_num):
        class_data_num.append(len(labels[labels == l]))
        if class_data_num[-1]==0:
            class_data_num[-1] = 1
    return class_data_num

centroids=centroids_cal(train_iter)

def calculate_closest(input):
    batch_size=input.size(0)
    features = textcnn.extract_feature(input)
    x_expand = features.unsqueeze(1).expand(-1, class_num, -1)
    centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
    dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
    _,min_cur_index = torch.min(dist_cur,dim=1)
    return min_cur_index

ouf = open(output_path,'w')
for batch in test_iter:
    inputs, target = batch.text, batch.label
    inputs = inputs.to(device)
    min_cur_index = calculate_closest(inputs)
    res = min_cur_index.detach().cpu().numpy().tolist()
    truth = target.numpy().tolist()
    for i in range(len(res)):
        ouf.write(str(res[i]) + "," + str(truth[i]) + "\n")
ouf.close()








