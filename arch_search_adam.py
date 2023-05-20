#!/usr/bin/env python
# coding: utf-8

# #### Import the libraries

# In[ ]:


import os
import random
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from transformers import BartTokenizer
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

from torch.autograd import Variable
from architect_adam import Architect
from BART import *
from attention_params import *
from ClassifierModel import *
from data_set import *


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_ = 9

seed_torch(seed_)


# ##### The arguments for the code

# # In[ ]:


# import sys,os,argparse
# from IPython.display import HTML
# CONFIG_FILE = '.config_ipynb'
# if os.path.isfile(CONFIG_FILE):
#     with open(CONFIG_FILE) as f:
#         sys.argv = f.read().split()
# else:
#     sys.argv = ['arch_search.py']



# In[ ]:


parser = argparse.ArgumentParser("BART")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--clf_batch_size', type=int, default=8, help='classifier batch size')
parser.add_argument('--bart_batch_size', type=int, default=1, help='BART batch size')

# BART
parser.add_argument('--bart_learning_rate', type=float, default = 1e-3, help='init learning rate')
parser.add_argument('--bart_learning_rate_min', type=float, default = 5e-4 , help='min learning rate')
parser.add_argument('--bart_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--bart_weight_decay', type=float, default=0, help='weight decay')#change to 1e-2 if needed

# Classifier
parser.add_argument('--clf_learning_rate', type=float, default = 0.001, help='init learning rate')
parser.add_argument('--clf_learning_rate_min', type=float, default = 0.001, help='min learning rate')
parser.add_argument('--eps', type=float, default=1e-08, help='eps')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--classifier_weight_decay', type=float, default=0, help='weight decay')

# Other hyperparameters
parser.add_argument('--begin_epoch', type=float, default=0, help='PC Method begin')
parser.add_argument('--stop_epoch', type=float, default=5, help='Stop training on the framework and just train on the classification task')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=seed_, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.25, help='portion of training data')
parser.add_argument('--A_learning_rate', type=float, default=3e-4, help='learning rate for A')
parser.add_argument('--A_weight_decay', type=float, default=1e-3, help='weight decay for A')
parser.add_argument('--lambda_par', type=float, default=0.85, help='augmented dataset ratio')
parser.add_argument('--max_length', type=int, default=50, help='Maximum length while generating with the Bart model')
args = parser.parse_args()



# In[ ]:


args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# ##### The number of classes

# In[ ]:


# Number of classification classes
TEXT_CLASSES = 2


# ##### Define the tokenizer

# In[ ]:


#####################################################################
# load the tokenizerss

# Load the BART tokenizer.
print('Loading BART tokenizer...')
bart_tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')

# Load the BART classifier tokenizer.
print('Loading classifier Tokenizer...')
classifier_tokenizer = bart_tokenizer

#####################################################################


# ##### Losses

# In[ ]:


# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input, attn, labels, bart_model, classifier):

    # convert input to the bart encodings
    # # use the generate approach
    summary_ids = bart_model.generate(input)
    
    bart_logits = bart_model(input, attn, target_ids = summary_ids, target_attn = torch.ones_like(summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    soft_idx, idx = torch.max(bart_logits, dim=-1, keepdims= True)
    
    # the gumbel soft max trick

    # though it looks trivial, the functionality is immense. We get the hard one-hot vectors as output
    # however it is differentiable with respect to the soft_ids i.e., the indices corresponding to max prob logits
    
    # summary_ids = torch.zeros_like(bart_logits).scatter_(-1, idx, 1.).float().detach() + soft_idx - soft_idx.detach()
    
    # try with the original summary ids
    summary_ids = torch.zeros_like(bart_logits).scatter_(-1, summary_ids.unsqueeze(-1), 1.).float().detach() + soft_idx - soft_idx.detach()

    # classifier
    _, loss = classifier.loss(summary_ids, labels)
    
    return loss



# In[ ]:


# the loss for the encoder and the decoder model 
# this takes into account the attention for all the datapoints for the encoder-decoder model
def CTG_loss(input_ids, target_ids, input_attn, target_attn, attn_idx, A, bart_model):
    
    # batch size
    batch_size = target_ids.shape[0]
    
    # target_sequence_length of the model
    target_sequence_length = target_ids.shape[1]
    
    # vocab size of the bart model
    # distill bart-6-6 uses tokenizer.vocab_size - 1
    vocab_size = bart_model.vocab_size - 1
    
    # loss function
    criterion = bart_model._criterion
    
    # attention weights of the batch
    attention_weights = A(attn_idx)
    
    # similar to the loss defined in the BART model hugging face conditional text generation
    
    # probability predictions
    logits = (bart_model(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)).logits
    
    loss_vec = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
    
    loss_vec = loss_vec.view(batch_size, target_sequence_length).mean(dim = 1)
    
    loss = torch.dot(attention_weights, loss_vec)
    
    scaling_factor = 1e3
    
    return scaling_factor*loss


# ##### Training function


# In[ ]:


softmax = torch.nn.Softmax(dim = -1)

# In[ ]:


def train(before_val_accuracy, epoch, train_classifier, valid_classifier, test_classifier, train_bart, model, student, architect, optimizer, optimizer_stud_adam, lr_bart, lr_clf):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    # for batch losses
    batch_loss_clf, batch_loss_bart, batch_count = 0, 0, 0


    for step, batch_classifier in enumerate(train_classifier):

        model.bart_model.train()
        
        student.train()
        
        #####################################################################################

        # Input and its attentions
        input = Variable(batch_classifier[0], requires_grad=False).cuda()
        input_attn_classifier = Variable(batch_classifier[1], requires_grad=False).cuda()
        # Number of datapoints
        n = input.size(0)
        # Label
        target = Variable(batch_classifier[2], requires_grad=False).cuda()
        
        #####################################################################################
        # valid input_valid, target_valid, valid_attn_classifier
        
        # get a random minibatch from the search queue with replacement
        valid_batch = next(iter(valid_classifier))
        input_valid           = Variable(valid_batch[0], requires_grad=False).cuda()
        valid_attn_classifier = Variable(valid_batch[1], requires_grad=False).cuda()
        target_valid          = Variable(valid_batch[2], requires_grad=False).cuda()
        
        #####################################################################################

        # get a random minibatch from the search queue with replacement
        batch_bart = next(iter(train_bart))
        # Input and its attentions
        input_text = Variable(batch_bart[0], requires_grad=False).cuda()
        input_attn = Variable(batch_bart[1], requires_grad=False).cuda()
        # Target and its attentions
        target_text = Variable(batch_bart[2], requires_grad=False).cuda()
        target_attn = Variable(batch_bart[3], requires_grad=False).cuda()
        # attention indices for CTG loss
        attn_idx = Variable(batch_bart[4], requires_grad=False).cuda()
        
        #####################################################################################

        # get input_attn and target_attn
        # pc darts implementation
        if args.begin_epoch <= epoch <= args.stop_epoch:
            
            architect.step(input, target, input_attn_classifier, input_valid, target_valid, valid_attn_classifier, 
                           input_text, target_text, input_attn, target_attn, attn_idx, lr_bart, lr_clf, optimizer, optimizer_stud_adam)
        
        # end the framework training and just train on the classifier task after the stop epoch
        if epoch <= args.stop_epoch:
            # update the T and G model
            # change it later since we need both T', G' and T, G to update
            optimizer.zero_grad()
            
            loss_bart = CTG_loss(input_text, target_text, input_attn, target_attn, attn_idx, architect.A, model)
            
            # store the batch loss
            batch_loss_bart += loss_bart.item()

            loss_bart.backward()
            
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            
            optimizer.step()

        # update the classifier model
        # change it later since we need both S' and S to update
        
        optimizer_stud_adam.zero_grad()
        
        # the training loss
        logits, loss_tr = student.loss(input, target)

        # classifier loss on augmented dataset
        loss_aug = calc_loss_aug(input, input_attn_classifier, target, model, student)
        
        loss_classifier = loss_tr + (args.lambda_par*loss_aug)
        
        # store for printing
        batch_loss_clf += loss_classifier.item()
        
        loss_classifier.backward()
        
        nn.utils.clip_grad_norm(student.parameters(), args.grad_clip)
        
        # update the classifier model
        optimizer_stud_adam.step()        

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
        
        objs.update(loss_classifier.item(), n)
        
        top1.update(prec1.item(), n)
        
        top5.update(prec5.item(), n)
        
        # count the batch
        batch_count += 1 
        
        if step % args.report_freq == 0:

            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            
            # print on the command line
            print(batch_loss_clf/batch_count, batch_loss_bart/batch_count)
            
            # for batch losses
            batch_loss_clf, batch_loss_bart, batch_count = 0, 0, 0
            
            val_accuracy, val_loss = infer(test_classifier, student)

            if val_accuracy > before_val_accuracy:

                before_val_accuracy = val_accuracy

                utils.save(student, os.path.join(args.save, 'Classifier_weights.pt'))

                utils.save(model, os.path.join(args.save, 'BART_weights.pt'))

                utils.save(architect.A, os.path.join(args.save, 'Attention_weights.pt'))

    return top1.avg, objs.avg, before_val_accuracy


# #### Define the main function

# In[ ]:

def main():
    
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # define the loss for BART
    criterion = nn.CrossEntropyLoss(ignore_index = bart_tokenizer.pad_token_id, reduction='none')
    criterion = criterion.cuda()

    # The loss for the XL Net is in inbuilt function from hugging face transformers

    # BART model creation which is the teacher producing augmented text dataset
    model = BART(criterion, bart_tokenizer)
    # model.load_state_dict(torch.load(os.path.join(args.save, 'BART_weights.pt')))
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # student classifier model creation
    
    # Hyperparameters
    embedding_dim = 300
    TEXT_CLASSES = 2
    hidden_size = 64
    
    # vocab size of the bart model
    # distill bart-6-6 uses tokenizer.vocab_size - 1
    student = ClassifierModel(classifier_tokenizer.vocab_size - 1, embedding_dim, TEXT_CLASSES, hidden_size)
    # student.load_state_dict(torch.load(os.path.join(args.save, 'Classifier_weights.pt')))
    student = student.cuda()
    
    ##################################################################################################################################################
    # try with different optimizers
    # 1
    ##################################################################################################################################################
    # optimizer for the BART model
    optimizer = torch.optim.SGD(model.parameters(),args.bart_learning_rate,momentum=args.bart_momentum,weight_decay=args.bart_weight_decay)
    ##################################################################################################################################################  
    # 2
    ##################################################################################################################################################
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.bart_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.bart_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters,args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    ################################################################################################################################################## 
    # 3
    ##################################################################################################################################################
    # optimizer for the student classifier model
    optimizer_stud_adam = torch.optim.Adam(student.parameters(), lr = args.clf_learning_rate)
    ##################################################################################################################################################  
    # 4
    ##################################################################################################################################################
    # no_decay = ['bias', 'LayerNorm.weight']
    
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in student.classifier.classification_head.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in student.classifier.classification_head.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters,args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    
    ##################################################################################################################################################
        
    ##################################################################################################################################################
    # Training datset for classifier
    # The DataLoader needs to know our batch size for training, so we specify it here.
    # For fine-tuning BART on a specific task, the authors recommend a batch size of 16 or 32.
    # get dataset
    train_data_classifier = get_classifierDataset(classifier_tokenizer)

    # train validation split
    num_train = len(train_data_classifier)

    indices = list(range(num_train))

    split = int(np.floor(args.train_portion * num_train))
    
    # change the NUM_WORKERS to 2
    # Create the DataLoader for our training set.
    # MLO-train
    train_classifier = DataLoader(train_data_classifier, batch_size=args.clf_batch_size, 
                                  sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=0)
    # MLO-validation
    valid_classifier = DataLoader(train_data_classifier, batch_size=args.clf_batch_size, 
                                  sampler=SubsetRandomSampler(indices[split:2*split]), pin_memory=True, num_workers=0)
    # Validation dataset
    test_classifier  = DataLoader(train_data_classifier, batch_size=args.clf_batch_size, 
                                  sampler=SubsetRandomSampler(indices[2*split:]), pin_memory=True, num_workers=0)

    print(len(indices[:split]))
    
    ##################################################################################################################################################
    # Training datset for BART  
    # load dataset
    train_data_bart = get_BartDataset(bart_tokenizer)

    # load the attention parameters
    attention_weights = attention_params(len(train_data_bart))
    # attention_weights.load_state_dict(torch.load(os.path.join(args.save, 'Attention_weights.pt')))
    attention_weights = attention_weights.cuda()

    # Create the DataLoader for our training set.
    train_bart = DataLoader(train_data_bart, sampler=RandomSampler(train_data_bart), 
                            batch_size=args.bart_batch_size, pin_memory=True, num_workers=0)
    ##################################################################################################################################################

    scheduler_bart = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.bart_learning_rate_min)
    
    lr_clf = args.clf_learning_rate
    
    architect = Architect(model, attention_weights, student, args)
    
    # used for running the model from the instance it was interrupted while running
    start_epoch = 0

    before_val_accuracy = 20
    
    for epoch in range(start_epoch, args.epochs):
        
        lr_bart = scheduler_bart.get_lr()[0]
        
        logging.info('epoch %d lr BART %e lr Clf %e', epoch, lr_bart, lr_clf)

        # training
        train_acc, train_obj, before_val_accuracy = train(before_val_accuracy, epoch, train_classifier, valid_classifier, test_classifier, train_bart, model, student, architect, optimizer, optimizer_stud_adam, lr_bart, lr_clf)
        
        scheduler_bart.step()

        # validation
        valid_acc, valid_obj = infer(test_classifier, student)
        
        # logging info
        logging.info('train_obj %f train_acc %f valid_obj %f valid_acc %f', train_obj, train_acc, valid_obj, valid_acc)
        
        if valid_acc > before_val_accuracy:
            
            before_val_accuracy = valid_acc
            
            utils.save(student, os.path.join(args.save, 'Classifier_weights.pt'))
            
            utils.save(model, os.path.join(args.save, 'BART_weights.pt'))
            
            utils.save(attention_weights, os.path.join(args.save, 'Attention_weights.pt'))

        # Note to the user: add all the information needed to analyse on the logfile
        
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Acc':^9}")
        
        print("-"*70)
        
        print(f"{epoch + 1:^7} | {train_obj:^7} | {train_acc:^12.6f} | {valid_acc:^12.6f}")
        
        if epoch % 5 == 0:
            
            # print the attention weights and inspect it
            
            logging.info(str(("Attention Weights: ",attention_weights.alpha)))
            
            
        if epoch >= args.begin_epoch: # changed this condition <= change it to >=
            
            # get a random minibatch from the search queue with replacement
            batch_bart = next(iter(train_bart))
            
            # Input and its attentions
            
            input_text = Variable(batch_bart[0], requires_grad=False).cuda()
            
            summary_ids = model.generate(input_text)
            
            logging.info(str([bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]))


# ##### Inference block

# In[ ]:



def infer(valid_queue, student):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    softmax = torch.nn.Softmax(-1)

    for step, batch_classifier in enumerate(valid_queue):
        
        student.eval()
        
        # Input and its attentions
        input = Variable(batch_classifier[0], requires_grad=False).cuda()
        # Attn
        input_attn_classifier = Variable(batch_classifier[1], requires_grad=False).cuda()
        # Label
        target = Variable(batch_classifier[2], requires_grad=False).cuda()
        # find the attentions input_attn_classifier

        # the training loss
        logits, loss = student.loss(input, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
        
        n = input.size(0)
        
        objs.update(loss.item(), n)
        
        top1.update(prec1.item(), n)
        
        top5.update(prec5.item(), n)

    return top1.avg, objs.avg


# ##### Main block

# In[ ]:


if __name__ == '__main__':
    main() 


# In[ ]:





# In[ ]:




