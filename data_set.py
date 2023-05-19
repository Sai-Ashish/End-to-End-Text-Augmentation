import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(9)

def get_BartDataset(tokenizer):
    # Load the dataset.
    dataset = load_dataset('cnn_dailymail', '3.0.0', split = 'test')
    
    num_points = 10
    # get the training data
    train_sentence = dataset['article'][:num_points]
    train_target = dataset['highlights'][:num_points]

    # attention indices for calculation of losses
    attn_idx = torch.arange(len(train_sentence))

    article_encoding = tokenizer(train_sentence, return_tensors='pt', padding=True, truncation = True)

    article_input_ids = article_encoding['input_ids']
    article_attention_mask = article_encoding['attention_mask']

    print("Input shape: ")
    print(article_input_ids.shape, article_attention_mask.shape)

    target_encoding = tokenizer(train_target, return_tensors='pt', padding=True, truncation = True)

    target_input_ids = target_encoding['input_ids']
    target_attention_mask = target_encoding['attention_mask']

    print("Target shape: ")
    print(target_input_ids.shape, target_attention_mask.shape)

    # turn to the tensordataset
    train_data = TensorDataset(article_input_ids, article_attention_mask, target_input_ids, target_attention_mask, attn_idx)

    return train_data


def get_classifierDataset(tokenizer):
    
    # Load the dataset into a pandas dataframe.
    dataset = load_dataset('glue', 'sst2')
    
    # Get the lists of sentences and their labels.
    Total_data = 10000
    percent_split = 0.75
    num_points = round(Total_data*percent_split*4)
    
    # get the training data
    batch = dataset['train']['sentence']
    labels = dataset['train']['label']
    
    # load the test data for model evaluation seperately.
    
    # shuffle the data
    data = list(zip(batch, labels))
    random.shuffle(data)
    batch, labels = zip(*data)
    
    # partition the data
    train_batch = batch[:num_points]
    train_labels = torch.tensor(labels[:num_points])
    
    # tokenizing the sentences
    seq_length = 150
    encoding = tokenizer(train_batch, return_tensors='pt', padding=True, truncation = True, max_length=seq_length)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # print the shapes
    print("Input shape: ")
    print(input_ids.shape, attention_mask.shape,train_labels.shape,train_labels)
    
    # turn to the tensordataset
    train_data = TensorDataset(input_ids, attention_mask, train_labels)
    
    return train_data



