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
projection_head_dimension = features
sequence_length = seq_length
num_classes = 2



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

save_path = './lc_encoder.pth'
word2Vec_model = gensim.models.Word2Vec.load("word2vec.model")
encoder = RNN(input_size, hidden_size, num_layers, projection_head_dimension, sequence_length)
encoder.load_state_dict(torch.load(save_path))
save_path = './lc_network.pth'
net = Network(input_size, hidden_size, num_classes)
net.load_state_dict(torch.load(save_path))
encoder.eval()
net.eval()

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

input_size = features
hidden_size = features
num_layers = 2
num_classes = features
sequence_length = seq_length
learning_rate = 0.005
batch_size = 100
num_epochs = 10



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

sec_encoder = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
optimizer = optim.Adam(sec_encoder.parameters(), lr=learning_rate)

def set_A(data, target, confidence, index):
  n=data.shape[0]
  m=n-1
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  confidence=confidence.detach().cpu().numpy()
  A=np.zeros([m,data.shape[1]], dtype=np.float32)
  Alabel=np.zeros(m, dtype=np.float32)
  Aconfidence=np.zeros([m,confidence.shape[1]], dtype=np.float32)
  j=0
  for i in range(n):
    if i != index:
      A[j]=data[i]
      Alabel[j]=target[i]
      Aconfidence[j]=confidence[i]
      j=j+1
  A=torch.from_numpy(A)
  Alabel=torch.from_numpy(Alabel)
  Aconfidence=torch.from_numpy(Aconfidence)
  return A, Alabel, Aconfidence

def set_B(A, Alabel, Aconfidence, label):
  n=A.shape[0]
  A=A.detach().cpu().numpy()
  Alabel=Alabel.detach().cpu().numpy()
  Aconfidence=Aconfidence.detach().cpu().numpy()
  count=0
  for i in range(n):
    if Alabel[i]==label:
      count=count+1

  B=np.zeros([count,A.shape[1]], dtype=np.float32)
  Blabel=np.zeros(count, dtype=np.float32)
  Bconfidence=np.zeros([count,Aconfidence.shape[1]], dtype=np.float32)

  count=0
  for i in range(n):
    if Alabel[i]==label:
      B[count]=A[i]
      Blabel[count]=Alabel[i]
      Bconfidence[count]=Aconfidence[i]
      count=count+1

  B=torch.from_numpy(B)
  Blabel=torch.from_numpy(Blabel)
  Bconfidence=torch.from_numpy(Bconfidence)
  return B, Blabel, Bconfidence

def loss_pair(query, A, B, index, alpha=1):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  numerator=torch.exp((cos(query.view(1,features),B[index,:].view(1,features)))/alpha)
  denom=0
  for k in range(A.shape[0]):
    denom=denom+torch.exp((cos(query.view(1,features),A[k,:].view(1,features)))/alpha)

  score=-torch.log(torch.div(numerator,denom))

  return score

def get_predicted_class_confidence(target, confidence, index):
  if target == 0:
    return confidence[index][0]
  else:
    return confidence[index][1]

def supervised_contrastive_loss(values, target, mal_values, confidence, mal_confidence):
  mal_confidence=torch.from_numpy(mal_confidence)
  confidence=torch.from_numpy(confidence)
  batch_loss=0
  for i in range(values.shape[0]):
    c_i = get_predicted_class_confidence(target[i], confidence, i)
    query=values[i]
    A, Alabel, Aconfidence = set_A(values, target, confidence, i)
    B, Blabel, Bconfidence = set_B(A, Alabel, Aconfidence, target[i])
    A = torch.cat((A, mal_values), 0)
    if target[i] == 1:
      B = torch.cat((B, mal_values), 0)
      Bconfidence = torch.cat((Bconfidence, mal_confidence), 0)
    loss=0
    for j in range(B.shape[0]):
      c_p = get_predicted_class_confidence(target[i], Bconfidence, j)
      loss=loss+((c_i*c_p)*loss_pair(query, A, B, j))
    if B.shape[0] > 0:
      loss=loss/B.shape[0]
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

def calculate_session_count(act_file):
  session_number=0
  Lines = act_file.readlines()
  for line in Lines:
    session_number=session_number+1

  return session_number

def select_malicious_sessions(data_vector, label, encoder, net):
  dataset = Seq_Dataset(data_vector,label)
  dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for batch_idx, (data, targets) in enumerate(dataloader):
    encode, _ = encoder(data)
  encode = encode.detach().numpy()
  data = data.detach().numpy()
  targets = targets.detach().numpy()
  dataset=linear_Dataset(encode,targets)
  dataloader= DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for x, y in dataloader:
    x = x.squeeze(1)
    scores = net(x)
    _, predictions = scores.max(1)

  count = 0
  for i in range(x.shape[0]):
    if predictions[i] == 1:
      count = count+1
  index = 0
  pred_label = np.zeros(count, dtype=np.float32)
  pred_data = np.zeros((count, seq_length, features))
  for i in range(x.shape[0]):
    if predictions[i] == 1:
      pred_data[index] = data[i]
      pred_label[index] = predictions[i]
      index = index+1

  pred_data = pred_data.astype(np.float32)
  pred_label = pred_label.astype(np.float32)

  return pred_data, pred_label

def construct_malicious_session_set(encoder, net):
  for root, dirs, files in os.walk("all_training_sessions"):
    for filename in files:
      with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
        session_count = calculate_session_count(act_file)


  for root, dirs, files in os.walk("all_training_sessions"):
    for filename in files:
      with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
        data_vector, _, label = make_dataset(act_file, session_count)
        sessions, pred_label = select_malicious_sessions(data_vector, label, encoder, net)

  return sessions, pred_label

def get_predicted_labels(data_vector, label, encoder, net):
  dataset = Seq_Dataset(data_vector,label)
  dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for batch_idx, (data, targets) in enumerate(dataloader):
    encode, _ = encoder(data)
  encode = encode.detach().numpy()
  targets = targets.detach().numpy()
  dataset=linear_Dataset(encode,targets)
  dataloader= DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for x, y in dataloader:
    x = x.squeeze(1)
    scores = net(x)
    _, predictions = scores.max(1)
  predictions = predictions.detach().cpu().numpy()

  return data_vector, predictions

def get_confidence_scores(data_vector, label, encoder, net):
  dataset = Seq_Dataset(data_vector,label)
  dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for batch_idx, (data, targets) in enumerate(dataloader):
    encode, _ = encoder(data)
  encode = encode.detach().numpy()
  targets = targets.detach().numpy()
  dataset=linear_Dataset(encode,targets)
  dataloader= DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for x, y in dataloader:
    x = x.squeeze(1)
    scores = net(x)
  scores = scores.detach().numpy()

  return data_vector, label, scores

def select_random_malicious_sessions_train(data, label, encoder, net, batch_size = 20):
  if data.shape[0]< batch_size:
    batch_size = data.shape[0]
  np.random.shuffle(data)
  data_vector = data[0:batch_size]
  label = label[0:batch_size]
  dataset = Seq_Dataset(data_vector,label)
  dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for batch_idx, (data, targets) in enumerate(dataloader):
    encode, _ = encoder(data)
  encode = encode.detach().numpy()
  targets = targets.detach().numpy()
  dataset=linear_Dataset(encode,targets)
  dataloader= DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
  for x, y in dataloader:
    x = x.squeeze(1)
    scores = net(x)
  scores = scores.detach().numpy()

  return  data_vector,  label, scores

def get_malicious_batch(dataloader, model):
  for batch_idx, (data, targets) in enumerate(dataloader):
    values, _ = model(data)
  return values

def train_model(dataloader, model, optimizer, malicious_dataloader, confidence, mal_confidence):
  for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    values, _ = model(data)
    mal_values =  get_malicious_batch(malicious_dataloader, model)
    loss = supervised_contrastive_loss(values, targets, mal_values, confidence, mal_confidence)
    loss.backward()
    optimizer.step()

  return loss

def train_dataset(seq_length, features, model, optimizer, encoder, net):
  session_count=batch_size
  malicious_data, malicious_label = construct_malicious_session_set(encoder, net)
  for epoch in range(num_epochs):
    for root, dirs, files in os.walk("training_set"):
      for filename in files:
        with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
          data_vector, _, label = make_dataset(act_file, session_count)
          data_vector, label = get_predicted_labels(data_vector, label, encoder, net)
          data_vector, label, confidence = get_confidence_scores(data_vector, label, encoder, net)
          dataset = Seq_Dataset(data_vector,label)
          dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
          malicious_data, malicious_label, mal_confidence = select_random_malicious_sessions_train(malicious_data, malicious_label, encoder, net)
          malicious_dataset = Seq_Dataset(malicious_data, malicious_label)
          malicious_dataloader = DataLoader(dataset=malicious_dataset, shuffle=False, batch_size=len(malicious_dataset))
          loss = train_model(dataloader, model, optimizer, malicious_dataloader, confidence, mal_confidence)

  return loss

loss = train_dataset(seq_length, features, sec_encoder, optimizer, encoder, net)
save_path = './encoder.pth'
torch.save(sec_encoder.state_dict(), save_path)
