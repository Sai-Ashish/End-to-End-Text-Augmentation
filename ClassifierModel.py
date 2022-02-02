import os
import random
import torch
import numpy as np
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(9)

class Embedding_(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding_, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).cuda()

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.embedding.weight)

class ClassifierModel(torch.nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim,
                 out_dim,
                 hidden_size,
                 dropout=0.5,
                 batch_first=True,
                 bidirectional=True
                ):
        super(ClassifierModel, self).__init__()

        # hyper parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # model functions
        # architecture same as EDA LSTM
        self.embedding = Embedding_(vocab_size, embedding_dim).requires_grad_()
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.lstm1 = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=batch_first, bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(128, 32, batch_first=batch_first, bidirectional=bidirectional)
        self.dense1 = torch.nn.Linear(64, 20)
        self.dense2 = torch.nn.Linear(20, out_dim)
        self.relu = torch.nn.ReLU(True)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # get embedding
        embedding_out = self.embedding(x)
        
        # get the first hidden layer weights
        h_lstm1, _ = self.lstm1(embedding_out)
        
        # get the dropout from 1st hiddenlayer
        h_lstm1 = self.dropout_layer(h_lstm1)
        
        # get the second hidden layer weights        
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # get the dropout from 2nd hiddenlayer        
        h_lstm2 = (self.dropout_layer)(h_lstm2)
        
        # get the last layer output for classification
        lstm2_hidden = h_lstm2[:, -1, :]
        
        # find the dense 
        dense_hidden = self.relu(self.dense1(lstm2_hidden))
        
        out = self.dense2(dense_hidden)
        
        return out

    def loss(self, x, target):

        logits = self(x)

        return logits, self.loss_fn(logits, target)

    # new model for the definitions of gradients in architec.py 
    def new(self):

        model_new = ClassifierModel(self.vocab_size, self.embedding_dim, self.out_dim, self.hidden_size).cuda()

        return model_new