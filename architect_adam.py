import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(9)

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


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

class Architect(object):

    def __init__(self, bart_model, A, classifier, args):

        self.max_length = args.max_length

        # classifier
        self.eps = args.eps
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.classifier_network_weight_decay = args.classifier_weight_decay

        # bart
        self.bart_network_momentum = args.bart_momentum
        self.bart_network_weight_decay = args.bart_weight_decay

        # bart model
        self.bart_model = bart_model
        self.encoder = bart_model.bart_model.model.encoder
        self.decoder = bart_model.bart_model.model.decoder

        # classifier model
        self.classifier = classifier
        
        self.lambda_par = args.lambda_par

        # importance parameter
        # import from the attention_params.py
        self.A = A

        # change to ctg dataset importance 
        # change to .parameters()
        self.optimizer = torch.optim.Adam(self.A.parameters(), 
          lr=args.A_learning_rate, betas=(0.5, 0.999), weight_decay=args.A_weight_decay)

  #########################################################################################
  # Computation of T' model named as unrolled model
  #   1) unrolled_T_model it calculated the one step gradient and updates the model
  #   2) _construct_T_model_from_theta convert the flattened theta to a model architecture
  # for 'T': Encoder model

  # Computation of 'G' model named as unrolled model
  #   1) unrolled_G_model it calculated the one step gradient and updates the model
  #   2) _construct_G_model_from_theta convert the flattened theta to a model architecture
  # for 'G': Decoder model

    def _compute_unrolled_bart_model(self, input, target, input_attn, target_attn, attn_idx, eta_bart, network_optimizer):
        # BART loss
        loss = CTG_loss(input, target, input_attn, target_attn, attn_idx, self.A, self.bart_model)
        # Unrolled encoder model
        theta_T = _concat(self.encoder.parameters()).data
        try:
            moment_T = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.encoder.parameters()).mul_(self.bart_network_momentum)
        except:
            moment_T = torch.zeros_like(theta_T)
        dtheta_T = _concat(torch.autograd.grad(loss, self.encoder.parameters(), retain_graph = True )).data + self.bart_network_weight_decay*theta_T
        # Unrolled decoder model
        theta_G = _concat(self.decoder.parameters()).data
        try:
            moment_G = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.decoder.parameters()).mul_(self.bart_network_momentum)
        except:
            moment_G = torch.zeros_like(theta_G)
        dtheta_G = _concat(torch.autograd.grad(loss, self.decoder.parameters())).data + self.bart_network_weight_decay*theta_G
        
        # convert to the model
        unrolled_bart_model = self._construct_bart_model_from_theta(theta_T.sub(eta_bart, moment_T+dtheta_T), theta_G.sub(eta_bart, moment_G+dtheta_G))
        return unrolled_bart_model

  # reshape the T model parameters
    def _construct_bart_model_from_theta(self, theta_T, theta_G):
    
        # create the new bart model
        bart_model_new = self.bart_model.new()

        # encoder update
        params, offset = {}, 0
        for k, v in self.encoder.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta_T[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta_T)
        bart_model_new.bart_model.model.encoder.load_state_dict(params)

        # Decoder update
        params, offset = {}, 0
        for k, v in self.decoder.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta_G[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta_G)
        bart_model_new.bart_model.model.decoder.load_state_dict(params)

        return bart_model_new

    # update the bart model with one step gradient update for unrolled model


#####################################################################################################
  # Computation of 'S' model named as unrolled model
  #   1) unrolled_S_model it calculated the one step gradient and updates the model
  #   2) _construct_S_model_from_theta convert the flattened theta to a model architecture
  # for 'S': Classifier model
  # the change is the loss which is sum of loss_tr and loss_aug with a lambda factor
    def _compute_unrolled_S_model(self, input, target, input_attn_classifier, eta_clf, unrolled_bart_model, classifier_optimizer):

        _, loss_tr = self.classifier.loss(input, target)

        # classifier loss on augmented dataset
        loss_aug = calc_loss_aug(input, input_attn_classifier, target, unrolled_bart_model, self.classifier)

        loss = loss_tr + (self.lambda_par*loss_aug)

        theta = _concat(self.classifier.parameters()).data
        
        grad = _concat(torch.autograd.grad(loss, self.classifier.parameters())).data
        
        try:
            # Exponential moving average of gradient values
            exp_avg = _concat(optimizer.state[v]['exp_avg'] for v in self.classifier.parameters())
            # Exponential moving average of squared gradient values
            exp_avg_sq = _concat(optimizer.state[v]['exp_avg_sq'] for v in self.classifier.parameters())
            step = _concat(torch.ones_like(optimizer.state[v]['exp_avg'])*optimizer.state[v]['step'] for v in self.classifier.parameters())
        
        except:
            step = torch.zeros_like(theta)
            # Exponential moving average of gradient values
            exp_avg = torch.zeros_like(theta)
            # Exponential moving average of squared gradient values
            exp_avg_sq = torch.zeros_like(theta)

        step = step + 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = exp_avg_sq.sqrt().add_(self.eps)
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        step_size = eta_clf * bias_correction2.sqrt() / bias_correction1
        if self.classifier_network_weight_decay != 0:
            theta.add_(theta, alpha=-self.classifier_network_weight_decay * eta_clf)

        unrolled_classifier_model = self._construct_S_model_from_theta(theta.sub((exp_avg/ denom)*step_size))
        
        return unrolled_classifier_model

    def _construct_S_model_from_theta(self, theta):
        
        classifier_new = self.classifier.new()

        params, offset = {}, 0
        for k, v in self.classifier.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        classifier_new.load_state_dict(params)

        return classifier_new.cuda()

  #########################################################################################

  # one step update for the importance parameter A
    def step(self, input_train, target_train, input_attn_classifier, input_valid, target_valid, valid_attn_classifier, input_text, target_text, input_attn, 
        target_attn, attn_idx, eta_bart, eta_clf, network_optimizer, classifier_optimizer):
        
        self.optimizer.zero_grad()

        unrolled_bart_model = self._compute_unrolled_bart_model(input_text, target_text, input_attn, target_attn, attn_idx, eta_bart, network_optimizer)

        unrolled_bart_model.bart_model.train()

        unrolled_classifier_model = self._compute_unrolled_S_model(input_train, target_train, input_attn_classifier, eta_clf, unrolled_bart_model, classifier_optimizer)

        _, unrolled_classifier_loss = unrolled_classifier_model.loss(input_valid, target_valid)

        unrolled_classifier_model.train()

        unrolled_classifier_loss.backward()

        vector_s_dash = [v.grad.data for v in unrolled_classifier_model.parameters()]

        implicit_grads_T, implicit_grads_G = self._outer(vector_s_dash, input_train, target_train, input_attn_classifier, input_text, target_text,
                                                                                 input_attn, target_attn, attn_idx, unrolled_bart_model, eta_bart, eta_clf)

        implicit_grads = implicit_grads_T + implicit_grads_G

        # change to ctg dataset importance
        # change to .parameters()
        for v, g in zip(self.A.parameters(), implicit_grads):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

        del unrolled_bart_model

        del unrolled_classifier_model


  ######################################################################
  # finite difference approximation of the hessian and the vector product for T
    def _hessian_vector_product_T(self, vector, input, target, input_attn, target_attn, attn_idx, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.encoder.parameters(), vector):
            p.data.add_(R, v)
        loss = CTG_loss(input, target, input_attn, target_attn, attn_idx, self.A, self.bart_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.encoder.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = CTG_loss(input, target, input_attn, target_attn, attn_idx, self.A, self.bart_model)

        # change to ctg dataset importance
        # change to .parameters()
        grads_n = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.encoder.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


  ######################################################################
  # finite difference approximation of the hessian and the vector product for G
    def _hessian_vector_product_G(self, vector, input, target, input_attn, target_attn, attn_idx, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.decoder.parameters(), vector):
            p.data.add_(R, v)
        loss = CTG_loss(input, target, input_attn, target_attn, attn_idx, self.A, self.bart_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.decoder.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = CTG_loss(input, target, input_attn, target_attn, attn_idx, self.A, self.bart_model)

        # change to ctg dataset importance
        grads_n = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.decoder.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  ######################################################################
  # function for the product of hessians and the vector product wrt T and function for the product of
  # hessians and the vector product wrt G
    def _outer(self, vector_s_dash, input_train, target_train, input_attn_classifier, input_text, target_text, input_attn, 
        target_attn, attn_idx, unrolled_bart_model, eta_bart, eta_clf, r=1e-2):
        
        R1 = r / _concat(vector_s_dash).norm()

        # plus S
        for p, v in zip(self.classifier.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        unrolled_bart_model.bart_model.train()

        # use the encoder 
        unrolled_T_model = unrolled_bart_model.encoder

        # use the decoder 
        unrolled_G_model = unrolled_bart_model.decoder

        loss_aug_p = calc_loss_aug(input_train, input_attn_classifier, target_train, unrolled_bart_model, self.classifier)

        # T
        vector_t_dash = torch.autograd.grad(loss_aug_p, unrolled_T_model.parameters(), retain_graph = True)

        grad_part1_T = self._hessian_vector_product_T(vector_t_dash, input_text, target_text, input_attn, target_attn, attn_idx)

        # G

        vector_g_dash = torch.autograd.grad(loss_aug_p, unrolled_G_model.parameters())

        grad_part1_G = self._hessian_vector_product_G(vector_g_dash, input_text, target_text, input_attn, target_attn, attn_idx)


        # minus S
        for p, v in zip(self.classifier.parameters(), vector_s_dash):
            p.data.sub_(2*R1, v)

        loss_aug_m = calc_loss_aug(input_train, input_attn_classifier, target_train, unrolled_bart_model, self.classifier)
        
        # T

        vector_t_dash = torch.autograd.grad(loss_aug_m, unrolled_T_model.parameters(), retain_graph = True)

        grad_part2_T = self._hessian_vector_product_T(vector_t_dash, input_text, target_text, input_attn, target_attn, attn_idx)

        # G

        vector_g_dash = torch.autograd.grad(loss_aug_m, unrolled_G_model.parameters())

        grad_part2_G = self._hessian_vector_product_G(vector_g_dash, input_text, target_text, input_attn, target_attn, attn_idx)

        for p, v in zip(self.classifier.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        grad_T = [(x-y).div_((2*R1)/(eta_bart*eta_clf*self.lambda_par)) for x, y in zip(grad_part1_T, grad_part2_T)]

        grad_G = [(x-y).div_((2*R1)/(eta_bart*eta_clf*self.lambda_par)) for x, y in zip(grad_part1_G, grad_part2_G)]

        return grad_T, grad_G