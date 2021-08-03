import os
import sys
import shutil
import sys
from tqdm import tqdm, tqdm_notebook
from math import log, sqrt, pi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from utils import *


def inference(opt, dset, model, flow):
    
    dset.set_mode('valid')
    model.eval()
    flow.eval()
    val_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True, collate_fn=pad_collate)
    pred = open('samples/questions.txt', "w")
    
    for batch_idx, batch in tqdm(enumerate(val_loader)):
        if batch_idx == 5:
            break

        model_inputs, labels, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l, device=opt.device, return_tokens = True)

        model_inputs = [flow] + model_inputs                
        model.get_conditional(*model_inputs)
        log_p, logdet, flow_z, inputs = model.encode(flow)
                            
        show_size = 32
        sentenses1 = get_sentences(model, model_inputs[1].contiguous().view(-1), dset, model_inputs[1].size(1), show_size, early_stop=False)
        
        sentenses2 = []
        if opt.use_ar:
            for i in range(show_size):
                sentenses2.append(beam.beam_search(flow, [flow.sample[0][i:i+1]], dset, i, beam_size=5, return_tokens=False, max_decode_steps=model.max_len-1, early_stop=False, cos=cos))
        reconstructed, logits = model.decode(flow, inputs, flow_z, True, show_size, cos)   
        sentenses3 = get_sentences(model, logits.view(-1, logits.size(-1)), dset, model.max_len, show_size, early_stop=False, show_all=True)    
        
        for i in range(len(sentenses1)):
            pred.write(sentenses1[i] + "\n")      
            pred.write(sentenses2[i] + "\n")      
            pred.write(sentenses3[i] + "\n")      
            pred.write("\n")      

            
    pred.close()
        
if __name__ == "__main__":

    torch.manual_seed(2020)
    
    opt = BaseOptions().parse()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpus[0])
    cudnn.benchmark = True
    
    from model.qg_flow import seqFlow        
    if opt.use_ar:
        from get_tokens_ar import *
        from model.flow.flowauto_eb import Glow
    else:       
        from get_tokens import *
        from model.flow.flownonauto import Glow
        
    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer
#     z_sample = [torch.randn([3,300,128,1]).cuda() * 0.7, torch.randn([3,300,64,1]).cuda() * 0.7, torch.randn([3,600,32,1]).cuda() * 0.7]
    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    model = seqFlow(opt)             
    model.load_embedding(dset.vocab_embedding)   
    model.cuda()
    
    flow = Glow(opt.flow_hidden, opt.flow_l, opt.flow_k, model.max_len, use_transformer=opt.use_transformer, use_recurrent=opt.use_recurrent, use_recurpling=opt.use_ar, squeeze_size=opt.squeeze_dim, embedding=model.embedding).cuda()
    cos = nn.CosineSimilarity(dim=2, eps=1e-12)

    if opt.restore_name:
        flow.load_state_dict(torch.load(opt.results_dir_base+opt.restore_name+'/best_valid_flow.pth', map_location='cuda:0'))


    flow.embedding = model.embedding
    flow.embedding.weight.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
    print('The number of parameters of model is', num_params, "M")
    num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)/1000000.0
    print('The number of parameters of flow is', num_params, "M")
        
    dset.return_tokens = True
    cur_loss = inference(opt, dset, model, flow)

