# Teacher model for the training of student model which is a text classifier in our case

import os
import random
import numpy as np
import copy
from transformers import BartForConditionalGeneration
import torch
import torch.nn as nn
import torch.nn.functional as F

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(9)

class BART(nn.Module):
    
    def __init__(self, criterion, tokenizer):
        super(BART, self).__init__()
        # few parameters needed to define the loss function and generate function
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        # loss type CrossEntropyLoss(ignore_index = self.pad_token_id)
        self._criterion = criterion

        # BART model definition with encoder and the decoder
        self.bart_model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6')
        self.encoder = self.bart_model.model.encoder
        self.decoder = self.bart_model.model.decoder

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):

        out = self.bart_model(input_ids, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)

        return out

    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 5, max_length=75):
        
        # beam search
        summary_ids = self.bart_model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, no_repeat_ngram_size = 2, length_penalty = 0.75, repetition_penalty = 1.2)
        
        ## sampling with top_p
        #summary_ids = self.bart_model.generate( input_ids = input_ids, num_beams = 1, max_length = max_length, top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, repetition_penalty = 1.2)

        return summary_ids

    # new model for the definitions of gradients in architec.py 
    def new(self):

        # there is embedding layer and the summarization head that we will not train on 
        # we just train on the encoder and the decoder weights 
        model_new = BART(self._criterion, self.tokenizer).cuda()
        
        # hence we deep copy all the weights and update the required ones
        # use the first order approximation for the summarization head
        # i.e, do not use unrolled model w.r.t to the summarization head parameters
        model_new.bart_model.load_state_dict(self.bart_model.state_dict())
        
        return model_new
