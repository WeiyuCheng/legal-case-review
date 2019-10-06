import os
import time
import sys
import torch
import torch.nn.functional as F
import torchtext.data as data

from textcnn_model import TextCNN
from textcnn_data_helper import process_data
from logger import Logger

lr=0.001
epoch_num=20
bs=96
dropout=0.5
embedding_dim=128
test_step=500
log_step = 10
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
filter_size = "2,3,4"
filter_num = 100
# the first model will be saved when acc > min_acc
min_acc = 80
# whether save the output and model
save=True
input_dir = "./data/"
model_dir = "./model/textcnn/"
log_dir = "./log/textcnn/%d/" % int(time.time())

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

logger = Logger(log_dir)


# set for tensorborad
def set_tensorboard(loss, train_acc, epoch, logger):
    info = {
        'loss': loss,
        'acc':train_acc
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # feature.data.t_(), target.data.sub_(1)
        feature, target = feature.to(device), target.to(device)
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy =  (100.0 * corrects.detach().cpu().item()) / (size*1.0)
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def save_model(model, save_dir,prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path=save_dir+prefix+".pkl"
    torch.save(model.state_dict(), save_path)


def test(textcnn, test_iter,output_dir):
    output_path = output_dir + "test.txt"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    textcnn.eval()

    corrects = 0
    # size = 0
    size = len(test_iter.dataset)


    ouf = open(output_path,'w')
    for batch in test_iter:
        # size+=1
        feature, target = batch.text, batch.label
        feature, target = feature.to(device), target.to(device)
        logits = textcnn(feature)
        res = torch.max(logits, 1)[1].view(target.size()).detach().cpu().numpy()
        truth = batch.label.numpy()
        for i in range(len(res)):
            ouf.write(str(res[i])+ ","+str(truth[i])+"\n")
        # if res==truth:
        #     corrects+=1
        corrects += np.sum(res==truth)

    ouf.close()
    acc = (corrects*1.0)/(size*1.0)
    print("\nTset Acc: %.6f" % acc)

if __name__ == '__main__':
    text_field = data.Field(lower=True, batch_first=True)
    label_field = data.Field(sequential=False, use_vocab=False, batch_first=True)
    train_iter, dev_iter, test_iter = process_data(text_field=text_field, label_field=label_field, data_dir=input_dir,
                                                   batch_size=bs)
    vocabulary_size = len(text_field.vocab)
    class_num = 183
    textcnn = TextCNN(vocabulary_size=vocabulary_size, class_num=class_num, filter_num=filter_num,
                      filter_sizes=filter_size, embedding_dim=embedding_dim, dropout=dropout)
    textcnn = textcnn.to(device)
    optimizer = torch.optim.Adam(textcnn.parameters(), lr=lr)
    textcnn.train()
    steps = 0
    best_acc = min_acc

    for epoch in range(1, epoch_num + 1):
        for batch in train_iter:
            textcnn.train()
            feature, target = batch.text, batch.label
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logits = textcnn(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % log_step == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = (100.0 * corrects.detach().cpu().item()) / (batch.batch_size * 1.0)
                print('\rEpoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                                           steps,
                                                                                           loss.item(),
                                                                                           train_acc,
                                                                                           corrects,
                                                                                           batch.batch_size))
            if steps % test_step == 0:
                dev_acc = eval(dev_iter, textcnn)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    prefix = "dim%d_filter_%s_num%d_acc%.4f" % (
                    embedding_dim, filter_size, filter_num, dev_acc)
                    if save:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save_model(textcnn, save_dir=model_dir, prefix=prefix)
                        output_dir = "./output/textcnn/%s/" % prefix
                        test(textcnn, test_iter, output_dir)