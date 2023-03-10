import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import LSTMTagger
from dataset import PosDataset
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
import random
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)
random.seed(0)
def train(dataset, model, args):
    print("Entered Training...")
    count=0
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    #use cross entropy loss
    loss_function = nn.NLLLoss()
    accuracy_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()
    optimizer = optim.SGD(model.parameters(), lr=0.8)
    for epoch in range(args.max_epochs):
        for batch, (sentence, tags) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            y_pred = model(sentence)
            loss = loss_function(y_pred, tags)
            loss.backward()
            optimizer.step()
            pred_list = list()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            accuary = (y_pred.argmax(1) == tags).float().mean()
            tags = tags.tolist()
            pred_list = pred_list
            recall = recall_score(tags, pred_list, average='macro', zero_division=0)
            precision = precision_score(tags, pred_list, average='macro', zero_division=0)
            f1 = f1_score(tags, pred_list, average='macro', zero_division=0)
            accuracy_list.append(accuary.item())
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            print({
                'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': accuary.item(), 'f1': f1})
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list), "avg f1: ", sum(f1_list)/len(f1_list))
              
def eval(args, model, val_dataset):
    print("entered eval")
    model.eval()
    avg_list = list()
    precision_list = list()
    recall_list = list()
    f1_list = list()
    loss_function = nn.NLLLoss()
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    state_h, state_c = model.init_state(args.sequence_length)
    for batch, (sentence, val_tags) in enumerate(val_dataloader):
            y_pred = model(sentence)
            pred_list = list()
            accuary = (y_pred.argmax(1) == val_tags).float().mean()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            avg_list.append(accuary.item())
            val_tags = val_tags.tolist()
            pred_list = pred_list   
            recall = recall_score(val_tags, pred_list, average='macro', zero_division=0)
            precision = precision_score(val_tags, pred_list, average='macro', zero_division=0)
            f1 = f1_score(val_tags, pred_list, average='macro', zero_division=0)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            print("validation: ", { 'batch': batch, 'acc': accuary.item(), 'recall': recall, 'precision': precision, 'f1': f1})
    print("avg acc: ", sum(avg_list)/len(avg_list), "avg recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list), "avg f1: ", sum(f1_list)/len(f1_list))
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="train")
parser.add_argument('--max-epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--sequence-length', type=int, default=3)
args = parser.parse_args()


dataset = PosDataset(args)
torch.save(dataset, "telugu_train.pt")
model = LSTMTagger(len(dataset.word_to_index), len(dataset.tag_to_index))
model = model.to(device)
train(dataset, model, args)
model.eval()
torch.save(model, "telugu_model.pt")
args.dataset = "dev"
val_dataset = PosDataset(args)
torch.save(val_dataset, "telugu_val.pt")
args.dataset = "test"
test_dataset = PosDataset(args)
torch.save(test_dataset, "telugu_test.pt")
