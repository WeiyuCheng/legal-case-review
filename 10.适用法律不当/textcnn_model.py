import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocabulary_size, class_num, filter_num=100, filter_sizes="3,4,5", embedding_dim=128,dropout=0.5):
        super(TextCNN, self).__init__()

        chanel_num = 1
        filter_sizes = [int(size) for size in filter_sizes.split(',')]

        vocabulary_size = vocabulary_size
        embedding_dimension = embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def extract_feature(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        return x
