import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.rnn import *


class LangFlow(nn.Module):
    def __init__(self, opt):
        super(LangFlow, self).__init__()
        self.opt = opt
        
        vocab_size = opt.vocab_size
        
        self.max_len = opt.max_len
        self.trainable_paddings = opt.trainable_paddings

        self.embedding = nn.Embedding(vocab_size, opt.flow_hidden)
        self.embedding.weight.requires_grad = False   
    
    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        
    def sample_z(self, mu, var):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        sample = (mu + torch.exp(var / 2) * eps).view(mu.size(0), 1, mu.size(1))
        return sample

    def sample(self, mean=0, std=0.0):
        sample = torch.empty(self.flow_z[-1].size()).normal_(mean=mean,std=std).cuda()
        return sample
    
    
    def get_conditional(self, flow, q, q_l=None, a0=None, a0_l=None, a1=None, a1_l=None, a2=None, a2_l=None, a3=None, a3_l=None, a4=None, a4_l=None, sub=None, sub_l=None, vcpt=None, vcpt_l=None, vid=None, vid_l=None):
        x_q = flow.embedding(q)
        if self.trainable_paddings:
            pad_indexes = (q==0)
            x_q[pad_indexes] = flow.flow_padding.repeat(x_q[pad_indexes].size(0), 1)
        return x_q, q_l

    def encode(self, flow, flow_inputs, inputs_length, use_condition=True, use_q=True):  
        
        flow_inputs = torch.cat([flow_inputs, flow.flow_padding.repeat(flow_inputs.size(0), max(self.max_len-flow_inputs.size(1), 0), 1)], 1)[:, :self.max_len]
        log_p, logdet, flow_z = flow(flow_inputs, inputs_length)
        
        return log_p, logdet, flow_z, flow_inputs
           
        
    def get_sample(self, flow_z, temperature=0.1, show_size=5):
        z_sample = []
        for z in flow_z:
            z_new = torch.empty(z.size()).normal_(mean=0,std=1.15)
            z_sample.append(z_new[:show_size].cuda())
        return z_sample
        
    def get_init(self, flow, embeddings, flow_z=None, reconstruct=False, show_size=3, cos=None):
        if not reconstruct:
            z_sample = self.get_sample(flow_z, show_size=show_size) 
            reconstructed = flow.reverse(z_sample, embeddings[:show_size], reconstruct=False, cos=cos)

        else:
            flow_inputs = [z[:show_size] for z in flow_z]
            reconstructed = flow.reverse(flow_inputs, embeddings[:show_size], reconstruct=True, cos=cos)

        return reconstructed
        
    def decode(self, flow, embeddings, flow_z, if_sample=False, show_size=3, cos=None):
        with torch.no_grad():
            reconstructed = self.get_init(flow, embeddings, flow_z, if_sample, show_size, cos=cos)

            logits = torch.cat([cos(reconstructed[ii:ii+1].view(-1, 1, reconstructed.size(-1)).cpu(), self.embedding.weight.unsqueeze(0).cpu()) for ii in range(reconstructed.size(0))], 0).view(reconstructed.size(0), reconstructed.size(1), -1).cuda()
            
            return reconstructed, logits



