import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from textcnn_model import TextCNN
from torch.nn.parameter import Parameter
import tqdm
import numpy as np
from torch.autograd.function import Function

class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


class OLTR_classifier(nn.Module):
    def __init__(self, feat_dim=300, num_classes=183):
        super(OLTR_classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

    def forward(self, x, centroids, *args):
        # storing direct feature
        direct_feature = x

        batch_size = x.size(0)
        feat_size = x.size(1)

        # set up visual memory
        x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids

        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        values_nn.clamp_(min=0.1,max=1000)
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x)
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x)
        concept_selector = concept_selector.tanh()
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature

        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]


class DiscCentroidsLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, centroids, cuda, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.device = torch.device('cuda:%d'%cuda if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.centroids = centroids
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)

        # calculate attracting loss

        feat = feat.view(batch_size, -1)
        # To check the dim of centroids and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        #  计算 direct feature 与自己label对应的centroids的距离
        loss_attract = self.disccentroidslossfunc(feat, label, self.centroids, batch_size_tensor).squeeze()

        # calculate repelling loss

        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat, self.centroids.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        # margin = 50.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1e6)

        # loss = loss_attract + 0.05 * loss_repel
        loss = loss_attract + 0.01 * loss_repel

        return loss


class DiscCentroidsLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centroids / batch_size, None


class OLTR_loss(nn.Module):
    def __init__(self, num_classes, feat_dim, featureloss_ratio, centroids, cuda, size_average=True):
        super(OLTR_loss,self).__init__()
        self.device = torch.device('cuda:%d' % cuda if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.feature_num = feat_dim
        self.featureloss_ratio = featureloss_ratio*1.0
        self.size_average = size_average
        self.featureloss = DiscCentroidsLoss(num_classes, feat_dim, centroids, cuda, size_average)
        self.performanceloss = nn.CrossEntropyLoss()


    def forward(self, logits, feature, label):
        self.feature_loss = self.featureloss(feature,label)
        self.performance_loss =  self.performanceloss(logits,label)
        return self.performance_loss + self.featureloss_ratio*self.feature_loss





class OLTR_For_Textcnn(nn.Module):
    def __init__(self, pretrained_model_path, vocabulary_size,filter_sizes,filter_num, data=None,train=False,cuda=1):
        super(OLTR_For_Textcnn,self).__init__()
        self.device = torch.device('cuda:%d'%cuda if torch.cuda.is_available() else 'cpu')
        self.textcnn = TextCNN(vocabulary_size=vocabulary_size, class_num=183, filter_num=filter_num,
                                filter_sizes=filter_sizes, embedding_dim=128)
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        self.textcnn.load_state_dict(checkpoint)
        self.textcnn = self.textcnn.to(self.device)
        # fix all param in textcnn when training OLTR
        for param_name, param in self.textcnn.named_parameters():
            param.requires_grad = False
        self.textcnn.eval()
        self.classes_num = 183
        self.feature_dim = len(filter_sizes.split(","))*filter_num

        self.classifier = OLTR_classifier(self.feature_dim,self.classes_num)

        self.centroids = nn.Parameter(torch.randn(self.classes_num, self.feature_dim))
        if train and data is not None:
            print("update centroid with data")
            self.centroids.data = self.centroids_cal(data)
        elif train and data is None:
            raise ValueError("Train mode should update centroid with data")
        else:
            print("Test mode should load pretrained centroid")



    def forward(self, x, *args):
        feature=self.textcnn.extract_feature(x)
        logits, _ = self.classifier(feature ,self.centroids)
        return logits,feature

    def class_count(self, data):
        labels = np.array([int(ex.label) for ex in data.dataset])
        class_data_num = []
        for l in range(self.classes_num):
            class_data_num.append(len(labels[labels == l]))
            if class_data_num[-1]==0:
                class_data_num[-1] = 1
        return class_data_num

    def centroids_cal(self, data):

        centroids = torch.zeros(self.classes_num, self.feature_dim).to(self.device)

        print('Calculating centroids.')

        # for model in self.networks.values():
        #     model.eval()
        self.textcnn.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):

            for batch in data:
                inputs, labels = batch.text, batch.label
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                features = self.textcnn.extract_feature(inputs)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += features[i]

        # Average summed features with class count
        centroids /= torch.Tensor(self.class_count(data)).float().unsqueeze(1).to(self.device)

        return centroids


