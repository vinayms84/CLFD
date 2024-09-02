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

def build_word_to_vector():
  data = []
  Label= []
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("all_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()

              for line in Lines:
                acts, _, label = line.split(';')
                temp = []
                session_count=session_count+1

                for act in acts.split(','):
                  temp.append(act.lower())

                data.append(temp)
                Label.append(label.strip('\n'))


  return data

features=50
seq_length=25
data=build_word_to_vector()
word2Vec_model = gensim.models.Word2Vec(
                    data,
                    sg=1,
                    vector_size=features,
                    min_count=1

                    )

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

input_size = features
hidden_size = features
num_layers = 2
projection_head_dimension = features
sequence_length = seq_length
learning_rate = 0.005
train_batch_size = 100
num_epochs = 10


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, projection_head_dimension, sequence_length):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).float()
    self.projection_head = nn.Linear(hidden_size, projection_head_dimension)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    rnn_out, _ = self.lstm(x, (h0,c0))
    encoding = torch.mean(rnn_out, axis = 1)
    projection_head_output = self.projection_head(encoding)

    return encoding, projection_head_output

net = RNN(input_size, hidden_size, num_layers, projection_head_dimension, sequence_length)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def loss_pair(query, A, B, index, alpha=1):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  numerator=torch.exp((cos(query.view(1,projection_head_dimension),B[index,:].view(1,projection_head_dimension)))/alpha)
  denom=0
  for k in range(A.shape[0]):
    denom=denom+torch.exp((cos(query.view(1,projection_head_dimension),A[k,:].view(1,projection_head_dimension)))/alpha)

  score=-torch.log(torch.div(numerator,denom))

  return score

def construct_loss_denominator_samples(data, index):
  n=data.shape[0]
  m=n-1
  data=data.detach().cpu().numpy()
  A=np.zeros([m,data.shape[1]], dtype=np.float32)
  j=0
  for i in range(n):
    if i != index:
      A[j]=data[i]
      j=j+1

  return torch.from_numpy(A)

def construct_positive_pair(A, index, flag):
  n=A.shape[0]
  A=A.detach().cpu().numpy()
  B=np.zeros([1,A.shape[1]], dtype=np.float32)
  count=0
  if flag == 0:
    B[count] = A[index+1]
  else:
    B[count] = A[index-1]

  return torch.from_numpy(B)

def SimCLR_loss(values):
  batch_loss=0
  for i in range(values.shape[0]):
    query = values[i]
    if i%2==0:
      flag = 0
    else:
      flag = 1
    A = construct_loss_denominator_samples(values, i)
    B = construct_positive_pair(values, i, flag)
    loss = 0
    for j in range(B.shape[0]):
      loss=loss+loss_pair(query, A, B, j)
    batch_loss=batch_loss+loss

  return batch_loss

def make_dataset(act_file, session_count):
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  temp_label_old= []
  session_number=0
  Lines = act_file.readlines()
  for line in Lines:
    acts, temp_label, label = line.split(';')
    temp_label_old.append(temp_label)
    session_label_old.append(label.strip('\n'))
    sequence_number=0
    for act in acts.split(','):
      if sequence_number<seq_length:
        x=word2Vec_model.wv.get_vector(act.lower())
        for i in range(features):
          dataset[session_number][sequence_number][i]=x[i]
        sequence_number=sequence_number+1
    session_number=session_number+1

  temp_label=np.array(temp_label_old)
  temp_label= temp_label.astype(np.float32)
  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, temp_label, session_label

def get_augmented_sessions(session):
  pivot = np.random.randint(low=0, high=session.shape[0]-4, size = 1)
  augmented_session_1 = session.copy()
  augmented_session_2 = session.copy()
  activity_1 = session[pivot].copy()
  activity_2 = session[pivot+1].copy()
  activity_3 = session[pivot+2].copy()

  index = np.random.permutation([pivot, pivot+1, pivot+2])

  augmented_session_1[index[0]] = activity_1
  augmented_session_1[index[1]] = activity_2
  augmented_session_1[index[2]] = activity_3

  index = np.random.permutation([pivot, pivot+1, pivot+2])

  augmented_session_2[index[0]] = activity_1
  augmented_session_2[index[1]] = activity_2
  augmented_session_2[index[2]] = activity_3

  return augmented_session_1, augmented_session_2

def construct_augmented_batch(data):
  data = data.detach().cpu().numpy()
  augmented_data =  np.zeros([2*data.shape[0], data.shape[1], data.shape[2]], dtype=np.float32)
  for i in range(data.shape[0]):
    augmented_session_1, augmented_session_2 = get_augmented_sessions(data[i])
    augmented_data[2*i] = augmented_session_1
    augmented_data[2*i+1] = augmented_session_2

  return torch.from_numpy(augmented_data)

def train_model(dataloader,net,optimizer,num_epochs):
  for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    augmented_data = construct_augmented_batch(data)
    _, head = net(augmented_data)
    loss =  SimCLR_loss(head)
    loss.backward()
    optimizer.step()

  return head

def train_dataset(seq_length,features, model,optimizer,batch_size):
  session_count=batch_size
  for epoch in range(num_epochs):
    for root, dirs, files in os.walk("training_set"):
      for filename in files:
        with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
          data_vector, temp_label, label = make_dataset(act_file, session_count)
          temp_label=torch.from_numpy(temp_label)
          dataset=Seq_Dataset(data_vector,label)
          dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
          loss = train_model(dataloader,model,optimizer,num_epochs)

  return loss

loss = train_dataset(seq_length, features, net, optimizer, train_batch_size)
save_path = './lc_encoder.pth'
torch.save(net.state_dict(), save_path)
word2Vec_model.save("word2vec.model")

