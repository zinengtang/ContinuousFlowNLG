import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import math
from model.rnn import *
logabs = lambda x: torch.log(torch.abs(x))

    

class embedding_linear(nn.Module):
    def __init__(self, in_channel, filter_size=1024):
        super(embedding_linear, self).__init__()
        
        self.net_e = nn.Sequential(
                nn.Linear(in_channel, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, in_channel*2)
            )  
        
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

        weight_r = np.random.randn(in_channel, in_channel)
        q_r, _ = la.qr(weight_r)
        w_p_r, w_l_r, w_u_r = la.lu(q_r.astype(np.float32))
        w_s_r = np.diag(w_u_r)
        w_u_r = np.triu(w_u_r, 1)
        u_mask_r = np.triu(np.ones_like(w_u_r), 1)
        l_mask_r = u_mask_r.T

        w_p_r = torch.from_numpy(w_p_r)
        w_l_r = torch.from_numpy(w_l_r)
        w_s_r = torch.from_numpy(w_s_r)
        w_u_r = torch.from_numpy(w_u_r)

        self.register_buffer('w_p_r', w_p_r)
        self.register_buffer('u_mask_r', torch.from_numpy(u_mask_r))
        self.register_buffer('l_mask_r', torch.from_numpy(l_mask_r))
        self.register_buffer('s_sign_r', torch.sign(w_s_r))
        self.register_buffer('l_eye_r', torch.eye(l_mask_r.shape[0]))
        self.w_l_r = nn.Parameter(w_l_r)
        self.w_s_r = nn.Parameter(logabs(w_s_r))
        self.w_u_r = nn.Parameter(w_u_r)
        
    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        weight_r = (
            self.w_p_r
            @ (self.w_l_r * self.l_mask_r + self.l_eye_r)
            @ ((self.w_u_r * self.u_mask_r) + torch.diag(self.s_sign_r * torch.exp(self.w_s_r)))
        )
        return weight, weight_r   
        
        
    def forward(self, coup_a, coup_b): 
        
        log, t = self.net_e(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)
        weight, weight_r = self.calc_weight() 
        
        out_b = (F.linear(F.linear(coup_b, weight), weight_r) + t) * s
                        
        logdet = torch.sum(torch.log(s).view(coup_a.shape[0], -1), 1) + torch.sum(self.w_s) + torch.sum(self.w_s_r)
        
        return out_b, logdet
    
    def reverse(self, coup_a, coup_b):
        weight, weight_r = self.calc_weight() 
        
        log, t = self.net_e(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)               
        in_b = F.linear(F.linear(coup_b / s - t, weight_r.inverse()), weight.inverse())
        
        return in_b


class lstm_linear(nn.Module):
    def __init__(self, in_channel, filter_size=1024):
        super(lstm_linear, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(in_channel, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, in_channel*2)
            )
        
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

        weight_r = np.random.randn(in_channel, in_channel)
        q_r, _ = la.qr(weight_r)
        w_p_r, w_l_r, w_u_r = la.lu(q_r.astype(np.float32))
        w_s_r = np.diag(w_u_r)
        w_u_r = np.triu(w_u_r, 1)
        u_mask_r = np.triu(np.ones_like(w_u_r), 1)
        l_mask_r = u_mask_r.T

        w_p_r = torch.from_numpy(w_p_r)
        w_l_r = torch.from_numpy(w_l_r)
        w_s_r = torch.from_numpy(w_s_r)
        w_u_r = torch.from_numpy(w_u_r)

        self.register_buffer('w_p_r', w_p_r)
        self.register_buffer('u_mask_r', torch.from_numpy(u_mask_r))
        self.register_buffer('l_mask_r', torch.from_numpy(l_mask_r))
        self.register_buffer('s_sign_r', torch.sign(w_s_r))
        self.register_buffer('l_eye_r', torch.eye(l_mask_r.shape[0]))
        self.w_l_r = nn.Parameter(w_l_r)
        self.w_s_r = nn.Parameter(logabs(w_s_r))
        self.w_u_r = nn.Parameter(w_u_r)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        weight_r = (
            self.w_p_r
            @ (self.w_l_r * self.l_mask_r + self.l_eye_r)
            @ ((self.w_u_r * self.u_mask_r) + torch.diag(self.s_sign_r * torch.exp(self.w_s_r)))
        )
        return weight, weight_r  
    
    def forward(self, coup_a, coup_b): 
        log, t = self.net(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)
        weight, weight_r = self.calc_weight() 
        out_b = ((F.linear(F.linear(coup_b, weight), weight_r)) + t) * s 
        
        logdet = torch.sum(torch.log(s).view(coup_a.shape[0], -1), 1) + torch.sum(self.w_s) + torch.sum(self.w_s_r)
        
        return out_b, logdet
    
    def reverse(self, coup_a, coup_b):
        log, t = self.net(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)     
        weight, weight_r = self.calc_weight() 
        in_b = F.linear(F.linear(coup_b / s - t, weight_r.inverse()), weight.inverse())
                
        return in_b

    
    
class RecurplingHead(nn.Module):
    
    
    def __init__(self, in_channel, embedding, filter_size=512, affine=True, use_transformer=True):
        super(RecurplingHead, self).__init__()

        self.num_layers = 32
        self.lstm_linear = lstm_linear(in_channel, filter_size)
        self.embedding_linear = embedding_linear(in_channel, filter_size)
        self.flow_lstm_r = RNNEncoder(in_channel, in_channel, bidirectional=False, dropout_p=0, n_layers=self.num_layers, rnn_type="lstm")
        
        self.embedding = embedding
        self.embedding.weight.require_grad = False
        self.reduce_layer = nn.Linear(in_channel*2, in_channel)
        self.context = nn.Parameter(torch.zeros([1, 1, in_channel]))
        self.h = nn.Parameter(torch.zeros([self.num_layers, 1, in_channel]))
        self.c = nn.Parameter(torch.zeros([self.num_layers, 1, in_channel]))
    
    def forward(self, inputs): 
        logdet = 0
        outputs = []
        prev_context = self.context.repeat(inputs.size(0), 1, 1)
        prev_states = (self.h.repeat(1, inputs.size(0), 1), self.c.repeat(1, inputs.size(0), 1))

        for i in range(1, inputs.size(1)):  
            
            prev_embedding = inputs[:, i-1:i, :]
            coup_b = inputs[:, i:i+1, :]      
            
            lstm_inputs = self.reduce_layer(torch.cat([prev_embedding, prev_context], 2))
            output, prev_states = self.flow_lstm_r(lstm_inputs, prev_states)

            out_b_e, logdet_e = self.embedding_linear(output, coup_b)
            prev_context =  out_b_e                                         
            out_b_l, logdet_l = self.lstm_linear(output, out_b_e)

            logdet += logdet_e + logdet_l
            outputs = outputs + [out_b_l] 
            
        outputs = [inputs[:, 0:1, :]] + outputs
        outputs = torch.cat(outputs, 1)               
        return outputs, logdet
    
    
    def reverse(self, inputs, cos=None):  
        
        prev_embedding = inputs[:, 0:1, :]
        outputs = [prev_embedding]        
        prev_context = self.context.repeat(inputs.size(0), 1, 1)
        prev_states = (self.h.repeat(1, inputs.size(0), 1), self.c.repeat(1, inputs.size(0), 1))
        
        for i in range(1, inputs.size(1)): 
            
            coup_b = inputs[:, i:i+1, :]  
            
            lstm_inputs = self.reduce_layer(torch.cat([prev_embedding, prev_context], 2))
            output, prev_states = self.flow_lstm_r(lstm_inputs, prev_states)
            
            out_b_l = self.lstm_linear.reverse(output, coup_b) 
            prev_context = out_b_l                      
            h_embed = self.embedding_linear.reverse(output, out_b_l)
            prev_embedding = h_embed
            outputs.append(prev_embedding)
            
        outputs = torch.cat(outputs, 1)
        return outputs

        
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))

    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output_linear(x)
        return x
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + sublayer(self.norm(x))

    
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, input_hidden, hidden, affine=True):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(6, input_hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=input_hidden, d_ff=hidden)
        self.input_sublayer = SublayerConnection(size=input_hidden)
        self.output_sublayer = SublayerConnection(size=input_hidden)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return x
      
        
class ActNorm(nn.Module):
    def __init__(self, in_channel, in_seqlen, logdet=True, transformer=True, recurrent=True):
        super().__init__()
        transformer = False
        recurrent = False
        self.transformer = transformer
        self.recurrent = recurrent
        self.loc = nn.Parameter(torch.zeros(1, 1, in_channel))
        self.scale = nn.Parameter(torch.ones(1, 1, in_channel))
        
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(2, 0, 1).contiguous().view(input.shape[2], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            std = (
                flatten.std(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-12))

    def forward(self, input):
        _, seq_length, _ = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = seq_length * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv1d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, seq_length, _ = input.shape

        out = F.linear(input, self.weight)
        logdet = (
            seq_length * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.linear(
            output, self.weight.squeeze().inverse()
        )


class InvConv1dLU(nn.Module):
    def __init__(self, in_seq):
        super().__init__()

        weight = np.random.randn(in_seq, in_seq)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, in_channel = input.shape

        weight = self.calc_weight()
        out = F.linear(input.transpose(2, 1), weight).transpose(2, 1)
        logdet = in_channel * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight
    
    def reverse(self, output):
        weight = self.calc_weight()

        return F.linear(output.transpose(2, 1), weight.inverse()).transpose(2, 1)


class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, 1, out_channel))

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True, use_transformer=True, use_recurrent=False):
        super().__init__()

        self.affine = affine
        self.use_recurrent = use_recurrent
        if use_recurrent:
            self.net = nn.Sequential(
                 FlowRNNEncoder(in_channel, in_channel, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm"),
                 ZeroLinear(in_channel*2, in_channel*2 if self.affine else in_channel // 2)                
            )
        
        elif use_transformer:
            self.net = nn.Sequential(
                TransformerBlock(in_channel, in_channel*2, self.affine),
                ZeroLinear(in_channel, in_channel*2 if self.affine else in_channel // 2)                
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(filter_size, filter_size, 1),
                nn.ReLU(inplace=True),
                ZeroLinear(filter_size, in_channel if self.affine else in_channel // 2),
            )

            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()

            self.net[2].weight.data.normal_(0, 0.05)
            self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            temp = self.net(in_a)
            log_s, t = temp.chunk(2, 2)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            temp = self.net(in_a)
            net_out = temp
            out_b = in_b + net_out
            logdet = None       
        outputs = torch.cat([in_a, out_b], 1)
        return outputs, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            temp = self.net(out_a)
            log_s, t = temp.chunk(2, 2)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            temp = self.net(out_a)
            net_out = temp
            in_b = out_b - net_out
            
        inputs = torch.cat([out_a, in_b], 1)
        return inputs

    
    
class Flow(nn.Module):
    def __init__(self, in_channel, in_seqlen, affine=True, conv_lu=True, use_transformer=False, use_recurrent=False):
        super().__init__()

        self.actnorm = ActNorm(in_channel, in_seqlen, transformer=use_transformer, recurrent=use_recurrent)

        if conv_lu:
            self.invconv = InvConv1dLU(in_seqlen)

        else:
            self.invconv = InvConv1d(in_seqlen)
        self.coupling = AffineCoupling(in_channel, affine=affine, use_transformer=use_transformer, use_recurrent=use_recurrent)

    def forward(self, input, inputs_length):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, use_transformer=False, in_seqlen=64, use_recurrent=False, use_recurpling=False, squeeze_size=2):
        super().__init__()
        self.squeeze_size = squeeze_size
        squeeze_dim = in_channel * self.squeeze_size

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, in_seqlen, affine=affine, conv_lu=conv_lu, use_transformer=use_transformer, use_recurrent=use_recurrent))

        self.split = split
        if split:
            self.prior = ZeroLinear(in_channel, in_channel * 2)

        else:
            self.prior = ZeroLinear(in_channel * self.squeeze_size, in_channel * self.squeeze_size * 2)

    def forward(self, input, inputs_length):

        logdet = 0
        b_size, seq_length, n_channel = input.shape
        squeezed = input.view(b_size, seq_length // self.squeeze_size, self.squeeze_size, n_channel)
        squeezed = squeezed.permute(0, 1, 3, 2).contiguous()
        out = squeezed.view(b_size, seq_length // self.squeeze_size, n_channel * self.squeeze_size)
        
        for flow in self.flows:
            out, det = flow(out, inputs_length)
            logdet = logdet + det
            
        if self.split:
            out, z_new = out.chunk(2, 2)
            mean, log_sd = self.prior(out).chunk(2, 2)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 2)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 2)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 2)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 2)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 2)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, seq_length, n_channel = input.shape

        unsqueezed = input.view(b_size, seq_length, n_channel // self.squeeze_size, self.squeeze_size)
        unsqueezed = unsqueezed.permute(0, 1, 3, 2).contiguous()
        unsqueezed = unsqueezed.view(
            b_size, seq_length * self.squeeze_size, n_channel // self.squeeze_size
        )
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, opt, embedding=None):
        in_channel, n_flow, n_block, in_seqlen, use_transformer, use_recurrent = opt.flow_hidden, opt.flow_l, opt.flow_k, opt.max_len, opt.use_transformer, opt.use_recurrent
        
        super().__init__()
        self.flow_padding = nn.Parameter(torch.zeros(in_channel))
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        self.in_seqlen = in_seqlen
        affine = True
        conv_lu=True
            
        for i in range(n_block - 1):
            if i == 0:
                i_squeeze = 1
                split=False
            else:
                i_squeeze = 2
                split=True
            self.in_seqlen //= i_squeeze
            
            self.blocks.append(Block(n_channel, n_flow, split=split, affine=affine, conv_lu=conv_lu, use_transformer=use_transformer, in_seqlen=self.in_seqlen, use_recurrent=use_recurrent, squeeze_size = i_squeeze))
            n_channel *= 1
        self.in_seqlen //= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, use_transformer=use_transformer, in_seqlen=self.in_seqlen, use_recurrent=use_recurrent, squeeze_size = 2))

    def forward(self, input, inputs_length):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
    
        for block in self.blocks:
            out, det, log_p, z_new = block(out, inputs_length)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        
        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, embeddings=None, reconstruct=False, cos=None):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        last_layer = input
        return input
