import sys
import os
import torch
import torch.nn.functional as F
import torchtext.data as data

from OLTR_model import OLTR_For_Textcnn,OLTR_loss
from textcnn_data_helper import process_data
import numpy as np


# hyper-parameter for model
cuda=1
device = torch.device('cuda:%d'%cuda if torch.cuda.is_available() else 'cpu')
load_model_dir = "./model/"
load_model_name = "pretrained_textcnn_dim128_filter_2,3,4_num100"
load_model_path = load_model_dir + load_model_name + ".pkl"
input_dir = "./data/"
model_dir = "./model/OLTR/"
filter_size = "2,3,4"
filter_num = 100


# hyper-parameter for training
bs=128
classifier_init_lr = 0.1
centroid_init_lr = 0.01
weight_decay = 0.0005
momentum = 0.9
scheduler_step=20
scheduler_gamma = 0.2
featureloss_ratio=0.1
min_acc=50
epoch_num=50
log_step=10
test_step=1000
save=True


def train(model,criterion,optimizer,scheduler,train_iter,dev_iter,test_iter):
    steps = 0
    best_acc = min_acc

    for epoch in range(1, epoch_num + 1):
        for batch in train_iter:
            model.train()
            input, target = batch.text, batch.label
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            logits,feature = model(input)
            loss = criterion(logits, feature, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % log_step == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = (100.0 * corrects.detach().cpu().item()) / (batch.batch_size*1.0)
                print('\rEpoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                                        steps,
                                                                                        loss.item(),
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        batch.batch_size))
            if steps % test_step == 0:
                dev_acc = eval(dev_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    prefix = "step%d_acc%.4f" % (steps, dev_acc)
                    if save:
                        print('Saving best model, acc: {:.4f}%'.format(best_acc))
                        save_model(model, save_dir=model_dir, prefix=prefix)
                        output_dir = "./output/OLTR/%s/" % prefix
                        test(model, test_iter, output_dir)
        scheduler.step()


def eval(data_iter, model):
    model.eval()
    corrects = 0
    for batch in data_iter:
        input, target = batch.text, batch.label
        input, target = input.to(device), target.to(device)
        logits, _ = model(input)
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    accuracy =  (100.0 * corrects.detach().cpu().item()) / (size*1.0)
    print('\rEvaluation - acc: {:.4f}%({}/{}) \n'.format(accuracy,
                                                        corrects,
                                                        size))
    return accuracy

def save_model(model, save_dir,prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path=save_dir+prefix+".pkl"
    torch.save(model.state_dict(), save_path)


def test(model, test_iter,output_dir):
    output_path = output_dir + "test.txt"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    model.eval()

    corrects = 0
    # size = 0
    size = len(test_iter.dataset)


    ouf = open(output_path,'w')
    for batch in test_iter:
        input, target = batch.text, batch.label
        input, target = input.to(device), target.to(device)
        logits, _ = model(input)
        res = torch.max(logits, 1)[1].view(target.size()).detach().cpu().numpy()
        truth = target.detach().cpu().numpy()
        for i in range(len(res)):
            ouf.write(str(res[i])+ ","+str(truth[i])+"\n")
        # if res==truth:
        #     corrects+=1
        corrects += np.sum(res==truth)

    ouf.close()
    acc = (corrects*1.0)/(size*1.0)
    print('\rTest - acc: {:.4f}%({}/{}) \n'.format(acc,
                                                    corrects,
                                                    size))




def init_model():
    text_field = data.Field(lower=True,batch_first=True)
    label_field = data.Field(sequential=False, use_vocab = False, batch_first=True)
    train_iter, dev_iter, test_iter = process_data(text_field=text_field, label_field=label_field, data_dir=input_dir,batch_size=bs)
    vocabulary_size = len(text_field.vocab)
    class_num = 183
    OLTR = OLTR_For_Textcnn(pretrained_model_path=load_model_path,vocabulary_size=vocabulary_size,data=train_iter,
                            filter_num=filter_num,filter_sizes=filter_size,cuda=cuda,train=True)
    OLTR = OLTR.to(device)
    OLTR_criterion = OLTR_loss(num_classes=class_num, feat_dim=OLTR.feature_dim, featureloss_ratio=featureloss_ratio, centroids=OLTR.centroids, cuda=cuda)
    optimizer = torch.optim.SGD([{'params':OLTR.classifier.parameters(),'lr':classifier_init_lr},
                                 {'params':OLTR.centroids,'lr':centroid_init_lr}],
                                momentum=momentum,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=scheduler_step,gamma=scheduler_gamma)
    return OLTR, OLTR_criterion, optimizer, scheduler, train_iter, dev_iter, test_iter


if __name__ == '__main__':
    OLTR, OLTR_criterion, optimizer, scheduler, train_iter, dev_iter, test_iter = init_model()
    train(OLTR, OLTR_criterion, optimizer, scheduler, train_iter, dev_iter, test_iter)







