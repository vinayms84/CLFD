import random
from torch.utils.data.sampler import Sampler
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import StandardScaler
from torch import linalg as LA
import numpy as np
import math
import gensim
import os
import nltk
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
nltk.download('punkt')
from sklearn.preprocessing import OneHotEncoder



seq_length=25
features=50
input_size = features
hidden_size = features
num_layers = 2
num_classes = 2
sequence_length = seq_length


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).float()



  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    rnn_out, _ = self.lstm(x, (h0,c0))
    out = torch.mean(rnn_out, axis = 1)


    return out, _

class Network(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Network, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.LeakyReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.l1(x)
    x = self.relu(x)
    x = self.l2(x)
    x = self.softmax(x)
    return x



word2Vec_model = gensim.models.Word2Vec.load("word2vec.model")
save_path = './encoder.pth'
sec_encoder = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
sec_encoder.load_state_dict(torch.load(save_path))
sec_encoder.eval()
save_path = './network.pth'
sec_net = Network(input_size, hidden_size, num_classes)
sec_net.load_state_dict(torch.load(save_path))

class Seq_Dataset(Dataset):
  def __init__(self, dataset, labels):

    labels=labels.flatten()
    self.x=torch.from_numpy(dataset[:,:,:])
    self.y=torch.from_numpy(labels).type(torch.LongTensor)
    self.n_samples=dataset.shape[0]


  def __getitem__(self,index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples

class linear_Dataset(Dataset):
  def __init__(self, dataset, labels):

    labels=labels.flatten()
    self.x=torch.from_numpy(dataset[:,:])
    self.y=torch.from_numpy(labels).type(torch.LongTensor)
    self.n_samples=dataset.shape[0]


  def __getitem__(self,index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples

def test_dataset(seq_length,features):
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()

              for line in Lines:

                session_count=session_count+1

  print('sesion_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
              for line in Lines:
                acts, _, label = line.split(';')
                session_label_old.append(label.strip('\n'))
                sequence_number=0
                for act in acts.split(','):
                  if sequence_number<seq_length:
                    x=word2Vec_model.wv.get_vector(act.lower())
                    for i in range(features):
                      dataset[session_number][sequence_number][i]=x[i]
                    sequence_number=sequence_number+1
                session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

data_vector, label= test_dataset(seq_length, features)
dataset=Seq_Dataset(data_vector,label)
dataloader= DataLoader(dataset=dataset, shuffle=True, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
  encode, _ = sec_encoder(data)
encode = encode.detach().numpy()
targets = targets.detach().numpy()
dataset=linear_Dataset(encode,targets)
dataloader= DataLoader(dataset=dataset, shuffle=True, batch_size=len(dataset))

def calculate_F1_FPR_score(loader, model):
    num_correct = 0
    num_samples = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    pred = []
    label =[]

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.squeeze(1)


            scores = model(x)
            _, predictions = scores.max(1)
            pred.append(predictions)
            label.append(y)
            tp += ((predictions == y) & (y==1)).sum()
            fp += ((predictions != y) & (y==0)).sum()
            fn += ((predictions != y) & (y==1)).sum()
            tn += ((predictions == y) & (y==0)).sum()



    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))
    fpr = fp/(tn+fp)
    model.train()
    return f1, fpr

def calculate_auc_scores(loader, model):
  with torch.no_grad():
    for x, y in loader:
      x = x.squeeze(1)
      scores = model(x)
      _, predictions = scores.max(1)

  y = y.detach().cpu().numpy()
  predictions = predictions.detach().cpu().numpy()
  roc_auc = roc_auc_score(y, predictions)

  return roc_auc

f1, fpr = calculate_F1_FPR_score(dataloader, sec_net)
print(f"f1, fpr: {f1, fpr}")
roc_auc = calculate_auc_scores(dataloader, sec_net)
print(f"auc_roc: {roc_auc}")
