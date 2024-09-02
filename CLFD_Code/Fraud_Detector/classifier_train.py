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

seq_length=25
features=50
input_size = features
hidden_size = features
num_layers = 2
sequence_length = seq_length
learning_rate = 0.005
train_batch_size = 100




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


save_path = './encoder.pth'
sec_encoder = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
sec_encoder.load_state_dict(torch.load(save_path))
sec_encoder.eval()

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

def create_fine_tune_dataset(seq_length,features):
  session_count=0
  for root, dirs, files in os.walk("fine_tune_data"):
    for filename in files:
      with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
        Lines = act_file.readlines()
        for line in Lines:
          session_count=session_count+1

  print('sesion_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("fine_tune_data"):
    for filename in files:
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

data_vector, label = create_fine_tune_dataset(seq_length, features)
data_vector, pred_label = get_predicted_labels(data_vector, label, encoder, net)
dataset = Seq_Dataset(data_vector, pred_label)
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(dataloader):
  encode, _ = sec_encoder(data)
encode = encode.detach().numpy()
targets = targets.detach().numpy()

dataset = linear_Dataset(encode,targets)
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=len(dataset))
sec_net = Network(input_size, hidden_size, num_classes)
secnet_optimizer = optim.Adam(sec_net.parameters(), lr=learning_rate)


def seperate_sessions_on_class(data, onehot_encode):
  data = data.detach().cpu().numpy()
  onehot_encode = onehot_encode.detach().cpu().numpy()
  normal_count = 0
  for i in range(data.shape[0]):
    if onehot_encode[i,0] == 1:
      normal_count = normal_count+1

  mal_count = data.shape[0]-normal_count
  normal_sessions = np.zeros((normal_count, features))
  normal_onehot_encode = np.zeros((normal_count, 2))
  malicious_sessions = np.zeros((mal_count, features))
  malicious_onehot_encode = np.zeros((mal_count, 2))
  count = 0
  for i in range(data.shape[0]):
    if onehot_encode[i,0] == 1:
      normal_sessions[count] = data[i]
      normal_onehot_encode[count] = onehot_encode[i]
      count = count+1

  count = 0
  for i in range(data.shape[0]):
    if onehot_encode[i,0] == 0:
      malicious_sessions[count] = data[i]
      malicious_onehot_encode[count] = onehot_encode[i]
      count = count+1




  return torch.from_numpy(normal_sessions), torch.from_numpy(normal_onehot_encode), torch.from_numpy(malicious_sessions), torch.from_numpy(malicious_onehot_encode)

def GCE_loss_sample(f, e, q=0.7):
  loss = e[0]*((1-torch.pow(f[0], q))/q)+e[1]*((1-torch.pow(f[1], q))/q)
  return loss

def GCE_loss(values, onehot_encode):
  batch_loss = 0
  for i in range(values.shape[0]):
    batch_loss =  batch_loss+GCE_loss_sample(values[i],onehot_encode[i])
  return batch_loss

def get_malicious_session_for_mixup(malicious_sessions, malicious_onehot_encode):
  index = np.random.randint(low=0, high=malicious_sessions.shape[0]-1, size = None)
  return malicious_sessions[index], malicious_onehot_encode[index]

def get_normal_session_for_mixup(normal_sessions, normal_onehot_encode):
  index = np.random.randint(low=0, high=normal_sessions.shape[0]-1, size = None)
  return normal_sessions[index], normal_onehot_encode[index]

def create_one_hot_encoding(target):
  target=target.detach().cpu().numpy()
  onehot_encoder = OneHotEncoder(sparse_output=False)
  integer_encoded = target.reshape(len(target), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  onehot_encoded = torch.from_numpy(onehot_encoded)
  return onehot_encoded


def train_loop_for_mini_batch(data, onehot_enocde, net, batch_size = train_batch_size):
  count = 0
  limit = data.shape[0]/batch_size
  while count < limit:
    secnet_optimizer.zero_grad()
    representation = data[count:count+batch_size]
    target = onehot_encode[count:count+batch_size]
    values = net(representation)
    loss = GCE_loss(values, target)
    loss.backward()
    secnet_optimizer.step()
    count = count+batch_size



def create_mixup_samples(data, onehot_encode, normal_sessions, normal_onehot_encode, malicious_sessions, malicious_onehot_encode, beta = 16):
  data = data.detach().cpu().numpy()
  onehot_encode = onehot_encode.detach().cpu().numpy()
  normal_sessions = normal_sessions.detach().cpu().numpy()
  normal_onehot_encode = normal_onehot_encode.detach().cpu().numpy()
  malicious_sessions = malicious_sessions.detach().cpu().numpy()
  malicious_onehot_encode = malicious_onehot_encode.detach().cpu().numpy()
  mixup_sessions = np.zeros([data.shape[0], features], dtype=np.float32)
  mixup_onehot_encode = np.zeros([data.shape[0], 2], dtype=np.float32)
  for i in range(data.shape[0]):
    index = onehot_encode[i,0]
    lamda = np.random.beta(beta, beta, size=None)
    if index == 0:
      X_j, e_j = get_malicious_session_for_mixup(malicious_sessions, malicious_onehot_encode)
      mixup_sessions[i] = (lamda*data[i])+((1-lamda)*X_j)
      mixup_onehot_encode[i] = (lamda*onehot_encode[i])+((1-lamda)*e_j)
    else:
      X_j, e_j = get_normal_session_for_mixup(normal_sessions, normal_onehot_encode)
      mixup_sessions[i] = (lamda*data[i])+((1-lamda)*X_j)
      mixup_onehot_encode[i] = (lamda*onehot_encode[i])+((1-lamda)*e_j)

  return torch.from_numpy(mixup_sessions), torch.from_numpy(mixup_onehot_encode)

num_epochs = 500
for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(dataloader):
    onehot_encode = create_one_hot_encoding(targets)
    normal_sessions, normal_onehot_encode, malicious_sessions, malicious_onehot_encode = seperate_sessions_on_class(data, onehot_encode)
    mixup_sessions, mixup_onehot_encode = create_mixup_samples(data, onehot_encode, normal_sessions, normal_onehot_encode, malicious_sessions, malicious_onehot_encode)
    train_loop_for_mini_batch(mixup_sessions, mixup_onehot_encode, net)
save_path = './network.pth'
torch.save(sec_net.state_dict(), save_path)

