import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rnn import *

class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None, xq_state=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.xq_state = xq_state
    def extend(self, token, log_prob, state, context=None, xq_state=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context,
                       xq_state=xq_state)
        return h
        
    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    
class BeamSearcher(object):
    def __init__(self,):
        pass
    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, model, model_inputs, dset, idx=0, beam_size=5, return_tokens=False, max_decode_steps=40, min_decode_steps=8, keep_beams=False, early_stop=True, expand_size=2, max_eos = 100):

        model_inputs = [temp[idx:idx+1] for temp in model_inputs[:12]]
        model.beam_size = beam_size     
        init_states, init_context = model.init_states, model.init_context
        h, c = init_states[0][:, idx], init_states[1][:, idx]  # [2, b, d] but b = 1

        hypotheses = [Hypothesis(tokens=[3*torch.ones([1, 1]).long().cuda()],
                                 log_probs=[0.0],
                                 state=(h, c),
                                 context=init_context[idx]) for _ in range(beam_size)]
        
        num_steps = 0
        results = []
        all_logits = []
        
        while num_steps < max_decode_steps and len(results) < beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            
            if isinstance(latest_tokens[0], int):
                prev_y = torch.LongTensor(latest_tokens).view(-1).cuda()
                prev_q = prev_y.view(-1, 1)
            else:
                prev_q = torch.cat(latest_tokens, 0)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)           
            
            logits, states, context_vector = model.decoder(prev_states, prev_context, prev_q, idx)
            if not early_stop:
                all_logits.append(logits)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, beam_size*expand_size, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]

                for j in range(beam_size*expand_size):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == 2 and early_stop:
                    if num_steps >= min_decode_steps:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == beam_size or len(results) == beam_size:
                    break
            num_steps += 1

        if len(results) == 0:
            results = hypotheses
        if len(results) < beam_size:
            results += self.sort_hypotheses(hypotheses)[:(beam_size-len(results))]
        h_sorted = self.sort_hypotheses(results)
        
        
        if keep_beams:
            sentences = []
            for final in h_sorted:
                sentence = ''
                last_pred_id = None
                all_eos = 0
                for pred_id in final.tokens[1:]:
                    if pred_id == 2:
                        all_eos += 1
                    if max_eos < 100:
                        if all_eos == max_eos or pred_id == 0:
                            sentence += dset.idx2word[pred_id] + ' '
                            break
                    elif (pred_id == 2 and early_stop) or pred_id == 0 or (last_pred_id == 2 and pred_id == 2): 
                        break
                    last_pred_id = pred_id
                    sentence += dset.idx2word[pred_id] + ' '
                    
                sentences+=[sentence]
        else:
            final = h_sorted[0]
            sentences = ''
            last_pred_id = None
            all_eos = 0
            for pred_id in final.tokens[1:]:  
                if pred_id == 2:
                    all_eos += 1
                if max_eos < 100:
                    if all_eos == max_eos:
                        sentence += dset.idx2word[pred_id] + ' '
                        break
                elif (pred_id == 2 and early_stop) or pred_id == 0 or (last_pred_id == 2 and pred_id == 2):
                    break
                last_pred_id = pred_id
                sentences += dset.idx2word[pred_id] + ' '
                
        if return_tokens:
            if len(final.tokens[1:]) < max_decode_steps:
                sentence_tokens = final.tokens[1:] + [0 for i in range(max_decode_steps - len(final.tokens[1:]))]
            else:
                sentence_tokens = final.tokens[1:max_decode_steps+1]
            return sentence_tokens

        else:
            return sentences

    
def get_ids(embedding, outputs):
    ids = torch.tensor([nn.CosineSimilarity(dim=-1)(embedding.weight, output).argmax() for output in outputs.contiguous().view(-1, 300)])
    ids = ids.view(outputs.size(0),outputs.size(1))
    return ids.numpy()


def get_idxs(dset, symbols, maxlen):
    sentences = []
    for symbols_os in symbols:
        sentence = []
        for pred_id in symbols_os.split(' '):
            if pred_id != '':
                sentence += [dset.word2idx[pred_id]]
        sentences += [sentence[:maxlen]]
    for i in range(len(sentences)):
        sentences[i] = sentences[i] + [0 for k in range(maxlen-len( sentences[i]))]
        
    return sentences


def get_words(dset, symbols):
    sentences = []
    for symbols_os in symbols:
        sentence = ''
        for pred_id in symbols_os:
            if pred_id == 2:
                break
            sentence += dset.idx2word[int(pred_id.data.cpu().numpy())] + ' '
        sentences += [sentence[:-1]]
    return sentences


def get_sentences(model, outputs, dset, sen_l, show_size=None, embedding=False, early_stop=True, show_all=False):
    sentences = []
    if embedding:
        pred_ids = get_ids(model.embedding, outputs)
    else:
        if len(outputs.size()) == 1:
            pred_ids = outputs.view(-1, sen_l).cpu().numpy()
        else:
            pred_ids = outputs.argmax(1).view(-1, sen_l).cpu().numpy()
    for pred_id_item in pred_ids:
        sentence = ''
        last_pred_id = None
        for pred_id in pred_id_item:
            if ((pred_id == 2 and early_stop) or pred_id == 0 or (last_pred_id == 2 and pred_id == 2)) and not show_all:
                break
            sentence += dset.idx2word[pred_id] + ' '
            last_pred_id = pred_id
        sentence = sentence[:-1]
        sentences.append(sentence)
    if show_size == None:
        show_size = len(sentences)
    return sentences[:show_size]

