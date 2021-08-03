import os
import sys
import shutil

from tqdm import tqdm, tqdm_notebook
from math import log, sqrt, pi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from config import *
sys.path.append('../')
# from cls import *

from loss import *
from utils import *
from dataset import TVQADataset, pad_collate, preprocess_inputs


def calc_loss(log_p, logdet, image_size, n_bins, input_hidden=64):
    n_pixel = input_hidden
# image_size * 
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def line_to_words(line):
    words = line.lower().split()
    words = [w for w in words if w != ","]
    words = [w if w != '<eos>' else '[SEP]' for w in words ]
    return words

def line_to_words_vcpt(vcpt_sentence):    
    attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
    unique_pairs = []
    for pair in attr_obj_pairs:
        if pair not in unique_pairs:
            unique_pairs.append(pair)
    words = []
    for pair in unique_pairs:
        words.extend(pair.split())
    return words


def train(opt, dset, model, flow, criterions, optimizer, epoch, previous_best_loss, schedular, tokenizer=None, bert=None, container1=None, container2=None, z_sample=None, cos=None):
    
    dset.set_mode('train')
    model.train()
    flow.train()
    
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=False, collate_fn=pad_collate)

    train_loss = []
    train_nll_loss = []
    train_recon_loss = []
    valid_loss_log = ["batch_idx\tloss\trecon-loss\tnll-loss\tvalid-loss\tvalid-recon-loss\tvalid-nll-loss"]
    
    torch.set_grad_enabled(True)
    
    beam = BeamSearcher()
    
    for batch_idx, batch in tqdm(enumerate(train_loader)):

        model_inputs, labels, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l, device=opt.device)


        model_inputs = [flow] + model_inputs
        
        flow_inputs, inputs_length = model.get_conditional(*model_inputs)
        log_p, logdet, flow_z, inputs = model.encode(flow, flow_inputs, inputs_length)
        recon_loss = 0

        nll_loss, log_p, log_det = calc_loss(log_p, logdet.mean(), inputs_length.float().mean(), 64, 300)
        
        loss = nll_loss
        
        optimizer[0].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer[0].step()    
        
        train_loss+=[loss.item()]
        train_nll_loss+=[nll_loss.item()]
        train_recon_loss+=[recon_loss]

        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx
            opt.writer.add_scalar("Train/Loss", np.mean(train_loss[-opt.log_freq:]), niter)
            opt.writer.add_scalar("Train/NLL-Loss", np.mean(train_nll_loss[-opt.log_freq:]), niter)
            opt.writer.add_scalar("Train/Recon-Loss", np.mean(train_recon_loss[-opt.log_freq:]), niter)
            
            valid_loss, valid_nll_loss, valid_recon_loss, val_logp, sentenses, recon_error = validate(opt, dset, model, flow, beam, mode="valid", cos=cos)
            opt.writer.add_scalar("Valid/Loss", valid_loss, niter)
            opt.writer.add_scalar("Valid/NLL-Loss", valid_nll_loss, niter)
            opt.writer.add_scalar("Valid/Recon-Loss", valid_recon_loss, niter)
            
            valid_log_str ="%02d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (batch_idx, loss, recon_loss, nll_loss, valid_loss, valid_recon_loss, valid_nll_loss)
            valid_loss_log.append(valid_log_str)
            if valid_loss < previous_best_loss:
                previous_best_loss = valid_loss                                                            
                torch.save(flow.state_dict(), os.path.join(opt.results_dir, "best_valid_flow.pth"))
                torch.save(optimizer[0].state_dict(), os.path.join(opt.results_dir, "optimizer1.pth"))
                
            print(" Train Epoch %d loss %.4f nll-loss %.4f recon-loss %.4f Val loss %.4f nll-loss %.4f recon-loss %.4f logp %.4f"
                      % (epoch, np.mean(train_loss[-opt.log_freq:]), np.mean(train_nll_loss[-opt.log_freq:]), np.mean(train_recon_loss[-opt.log_freq:]), valid_loss,  valid_nll_loss, valid_recon_loss, val_logp))
                
            for i in range(len(sentenses)):                
                print(sentenses[i])       
       
            torch.set_grad_enabled(True)
            model.train()
            flow.train()
            dset.set_mode("train")

        if opt.debug:
            break

    with open(os.path.join(opt.results_dir, "valid_loss.log"), "a") as f:
        f.write("\n".join(valid_loss_log) + "\n")

    return previous_best_loss


def validate(opt, dset, model, flow, beam, mode="valid", cos=None):
    
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    flow.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    valid_loss = []
    valid_nll_loss = []
    valid_recon_loss = []
    val_logp = []
  
    for k, batch in enumerate(valid_loader):
        
        model_inputs, labels, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l, device=opt.device)
       
        model_inputs = [flow] + model_inputs
        
        
        flow_inputs, inputs_length = model.get_conditional(*model_inputs)
        log_p, logdet, flow_z, inputs = model.encode(flow, flow_inputs, inputs_length)

        if k == 0:
            reconstructed, logits = model.decode(flow, inputs, flow_z, True, 1, cos) 
            recon_error = torch.abs(reconstructed[0][:model_inputs[2][0]] - inputs[0][:model_inputs[2][0]]).mean()
            print('reconstruction error:', recon_error)
            if recon_error > 2.0:
                print('----warning, invertibility_test failed----')
            print(logits[0].argmax(-1), model_inputs[1][0][1:])  
        recon_loss = 0
                             
        nll_loss, log_p, log_det = calc_loss(log_p, logdet.mean(), model_inputs[12].float().mean(), 1, 300)
        loss = nll_loss

        val_logp+=[log_p.item()]
        valid_loss+=[loss.item()]
        valid_nll_loss+=[nll_loss.item()]
        valid_recon_loss+=[recon_loss]


        if k == 0:
            show_size = 5            
            if opt.use_ar:
                z_sample = model.get_sample(flow_z, show_size=show_size)    
                sentenses = []
                for i in range(show_size):
                    sentenses.append(beam.beam_search(flow, flow_inputs[:1, :2], inputs_length[:1], [z_sample[0][i:i+1]], dset, i, beam_size=opt.beam_size, return_tokens=False, max_decode_steps=model.max_len-1, early_stop=False, cos=cos))
                sentenses_greedy = []
                for i in range(show_size):
                    sentenses_greedy.append(beam.beam_search(flow, flow_inputs[:1, :2], inputs_length[:1], [z_sample[0][i:i+1]], dset, i, beam_size=1, return_tokens=False, max_decode_steps=model.max_len-1, early_stop=False, cos=cos))
            else:
                reconstructed, logits = model.decode(flow, inputs, flow_z, False, show_size, cos)                   
                sentenses = get_sentences(model, logits.view(-1, logits.size(-1)), dset, model.max_len, show_size, early_stop=False, show_all=True)    
        if opt.debug:
            break

        if opt.val_steps:
            if opt.val_steps <= k+1:
                break
   
    valid_loss = np.mean(valid_loss)
    valid_nll_loss = np.mean(valid_nll_loss)
    valid_recon_loss = np.mean(valid_recon_loss)
    val_logp = np.mean(val_logp)
    
    return valid_loss, valid_nll_loss, valid_recon_loss, val_logp, sentenses, recon_error


if __name__ == "__main__":
    save_numpy = []
    save_numpy_index = []
        
    torch.manual_seed(2020)
    
    opt = BaseOptions().parse()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpus[0])
    from model.flow_entry import LangFlow  
    
    if opt.use_ar:
        from get_tokens_ar import *
        from model.flow.flowauto_eb4 import Glow
    else:       
        from get_tokens import *
        from model.flow.flownonauto import Glow
        
    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer
    z_sample = None
    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    model = LangFlow(opt)             
    model.load_embedding(dset.vocab_embedding)   
    model.cuda()
    
    flow = Glow(opt, embedding=model.embedding).cuda()
    cos = nn.CosineSimilarity(dim=2, eps=1e-12)
    flow.embedding = model.embedding
    flow.embedding.weight.requires_grad = False

    if opt.restore_name:
        flow.load_state_dict(torch.load(opt.results_dir_base+opt.restore_name+'/best_valid_flow.pth', map_location='cuda:0'), strict=False)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
    print('The number of parameters of model is', num_params, "M")
    num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)/1000000.0
    print('The number of parameters of flow is', num_params, "M")
    
    criterions = [EntropyLoss().cuda(), NLL().cuda()]
    optimizer = [torch.optim.Adam(filter(lambda p: p.requires_grad, flow.parameters()), lr=opt.lr)]

    if opt.restore_name:
        di=opt.results_dir_base+opt.restore_name+'/optimizer1.pth'
        optimizer[0].load_state_dict(torch.load(di, map_location='cuda:0'))    
        
    schedular = None
    best_loss = 1e10
    early_stopping_cnt = 0
    early_stopping_flag = False

    for epoch in range(opt.n_epoch):
        if not early_stopping_flag:
            # train for one epoch, valid per n batches, save the log and the best model
            cur_loss = train(opt, dset, model, flow, criterions, optimizer, epoch, best_loss, schedular, container1=save_numpy, container2=save_numpy_index, z_sample=z_sample, cos=cos)
            # remember best acc
            best_loss = min(cur_loss, best_loss)

        else:
            print("early stop with valid loss %.4f" % best_loss)
            opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            opt.writer.close()
            break  # early stop break

        if opt.debug:
            break

