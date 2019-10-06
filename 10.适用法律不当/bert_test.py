# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for classification.
"""
import os
import csv
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="model/google_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--load_model", default="pretrained_bert", type=str,
                        help="Path of the load model.")
    parser.add_argument("--load_model_dir", default="./model", type=str,
                        help="Path of the load model.")
    parser.add_argument("--vocab_path", default="./model/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--test_path", type=str, default="./data/test.tsv",
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./model/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="model/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    args.labels_num=183
    args.load_model_path="%s/%s.bin"%(args.load_model_dir, args.load_model)
    output_dir="./output/BERT/%s/"%args.load_model
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    args.output_path=output_dir+"test.txt"


    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.load_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read dataset.
    def read_dataset_law(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            # lines = f.readlines()
            f=csv.reader(f, delimiter='\t')
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                try:
                    # line = line.strip().split('\t')
                    if len(line) == 3:
                        label = int(line[2])
                        text = line[1]
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    else:
                        pass
                        
                except:
                    pass
        return dataset




    # Evaluation function.
    def evaluate(args):
        is_test=True
        dataset = read_dataset_law(args.test_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        print("The number of evaluation instances: ", instances_num)
        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        ouf=open(args.output_path,"w")

        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                    ouf.write(str(pred[j].item()) + "," + str(gold[j].item()) + "\n")
                correct += torch.sum(pred == gold).item()
        
            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")
            for i in range(confusion.size()[0]):
                if confusion[i,i].item()==0:
                    p,r,f1 = 0.0,0.0,0.0
                else:
                    p = confusion[i,i].item()/confusion[i,:].sum().item()
                    r = confusion[i,i].item()/confusion[:,i].sum().item()
                    f1 = 2*p*r / (p+r)
                if is_test:
                    print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)
        
            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][3]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))


            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][3])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print(MRR)
            return MRR

    evaluate(args)




if __name__ == "__main__":
    main()
