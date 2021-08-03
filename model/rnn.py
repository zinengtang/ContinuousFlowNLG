import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class FlowRNNEncoder(nn.Module):
    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm", return_hidden=True, return_outputs=True):
        super(FlowRNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        # - add return_hidden keyword arg to reduce computation if hidden is not needed.
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        if bidirectional:
            for a in ['weight_ih_l0', 'weight_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse']:
                torch.nn.init.xavier_normal(self.rnn.state_dict()[a])
        else:
            for a in ['weight_ih_l0', 'weight_hh_l0']:
                torch.nn.init.xavier_normal(self.rnn.state_dict()[a])            
    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, states=None, lengths=None):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        if lengths is None:
            varied_length=False
        else:
            varied_length=True
        if varied_length:
            sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
            packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
            outputs, hidden = self.rnn(packed_inputs)
        else:
            if states is None:
                outputs, hidden = self.rnn(inputs)
            else:
                outputs, hidden = self.rnn(inputs, states)
        
        if varied_length:
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
#         if self.return_ne
        return outputs

class RNNEncoder(nn.Module):
    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm", return_hidden=True, return_outputs=True):
        super(RNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        # - add return_hidden keyword arg to reduce computation if hidden is not needed.
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        if bidirectional:
            for a in ['weight_ih_l0', 'weight_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse']:
                torch.nn.init.xavier_normal(self.rnn.state_dict()[a])
        else:
            for a in ['weight_ih_l0', 'weight_hh_l0']:
                torch.nn.init.xavier_normal(self.rnn.state_dict()[a])            
    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, states=None, lengths=None):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        if lengths is None:
            varied_length=False
        else:
            varied_length=True
        if varied_length:
            sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
            packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
            outputs, hidden = self.rnn(packed_inputs)
        else:
            if states is None:
                outputs, hidden = self.rnn(inputs)
            else:
                outputs, hidden = self.rnn(inputs, states)
        
        if varied_length:
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
#         if self.return_hine
        return outputs, hidden


def max_along_time(outputs, lengths=None):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :return: (B, D)
    """
    if lengths is None:
        outputs = [outputs[i, ].max(dim=0)[0] for i in range(outputs.size(0))]
    else:
        outputs = [outputs[i, :int(lengths[i])].max(dim=0)[0] for i in range(len(lengths))]
    return torch.stack(outputs, dim=0)

def min_along_time(outputs, lengths):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :return: (B, D)
    """
    outputs = [outputs[i, :int(lengths[i]), :].min(dim=0)[0] for i in range(len(lengths))]
    return torch.stack(outputs, dim=0)

def mean_along_time(outputs, lengths=None):
    """ Get mean responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :return: (B, D)
    """
    if lengths is None:
        outputs = [outputs[i, ].mean(dim=0) for i in range(outputs.size(0))]
    else:
        outputs = [outputs[i, :int(lengths[i]), :].mean(dim=0) for i in range(len(lengths))]
    return torch.stack(outputs, dim=0)

def median_along_time(outputs, lengths=None):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :return: (B, D)
    """
    if lengths is None:
        outputs = [outputs[i, ].median(dim=0)[0] for i in range(outputs.size(0))]
    else:
        outputs = [outputs[i, :int(lengths[i])].median(dim=0)[0] for i in range(len(lengths))]
    return torch.stack(outputs, dim=0)