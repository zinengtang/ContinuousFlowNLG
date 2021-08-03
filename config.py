import os
import time
import torch
import argparse
from utils import mkdirp, load_json, save_json, save_json_pretty


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.parser.add_argument("--gpus", type=int, nargs='+', default=[2], help="")
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default="results/results_")
        self.parser.add_argument("--log_freq", type=int, default=1000, help="print, save training info")
        self.parser.add_argument("--lr", type=float, default=8e-4, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=3, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=100, help="mini-batch size for testing")
        self.parser.add_argument("--device", type=str, default='cuda', help="gpu ordinal, -1 indicates cpu")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--word_count_threshold", type=int, default=2, help="word vocabulary threshold")

        # model config
        self.parser.add_argument("--no_glove", action="store_true", help="not use glove vectors")
        self.parser.add_argument("--no_ts", action="store_true", help="no timestep annotation, use full length feature")
        self.parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet"], default=["vcpt","sub"], help="input streams for the model, will use both `vcpt` and `sub` streams")
        self.parser.add_argument("--n_layers_cls", type=int, default=1, help="number of layers in classifier")
        self.parser.add_argument("--flow_k", type=int, default=3, help="hidden size for the second lstm")
        self.parser.add_argument("--flow_l", type=int, default=6, help="hidden size for the second lstm")
        self.parser.add_argument("--flow_hidden", type=int, default=None, help="hidden size for flow")
        self.parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
        self.parser.add_argument("--max_len", type=int, default=16, help="max length for flow inputs")
        self.parser.add_argument("--max_sub_l", type=int, default=512, help="max length for subtitle")
        self.parser.add_argument("--max_vcpt_l", type=int, default=512, help="max length for visual concepts")
        self.parser.add_argument("--max_vid_l", type=int, default=480, help="max length for video feature")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        self.parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
        self.parser.add_argument("--use_baseline", action="store_true")
        
        self.parser.add_argument("--beam_size", type=int, default=5, help="beam size to autoregressive flow decoding")
        self.parser.add_argument("--return_beams", action="store_true")
        
        self.parser.add_argument("--use_transformer", action="store_true")
        self.parser.add_argument("--use_recurrent", action="store_true")
        self.parser.add_argument("--use_ar", action="store_true", help="whether to use autoregressive flow or non autoregressive")
        self.parser.add_argument("--share_weights", action="store_true")
        self.parser.add_argument("--arcoupling_linear_map", action="store_true")
        self.parser.add_argument("--trainable_paddings", action="store_true")
        self.parser.add_argument("--multiscale", action="store_true")
        self.parser.add_argument("--squeeze_dim", type=int, default=2, help="times to squeeze for multi-scale architecture in non autoregressive flow")
        
        self.parser.add_argument("--restore_name", type=str)
        self.parser.add_argument("--save_name", type=str)
        self.parser.add_argument("--val_steps", type=int,)
        # path config
        self.parser.add_argument("--train_path", type=str, default="./data/tvqa_train_processed.json",
                                 help="train set path")
        self.parser.add_argument("--valid_path", type=str, default="./data/tvqa_val_processed.json",
                                 help="valid set path")
        self.parser.add_argument("--test_path", type=str, default="./data/tvqa_test_public_processed.json",
                                 help="test set path")
        self.parser.add_argument("--glove_path", type=str, default="./data/glove.840B.300d.txt",
                                 help="GloVe pretrained vector path")
        self.parser.add_argument("--vcpt_path", type=str, default="./data/det_visual_concepts_hq.pickle",
                                 help="visual concepts feature path")
        self.parser.add_argument("--vid_feat_path", type=str, default="./data/tvqa_imagenet_pool5.h5",
                                 help="imagenet feature path")
        self.parser.add_argument("--vid_feat_size", type=int, default=2048,
                                 help="visual feature dimension")
        self.parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.pickle",
                                 help="word2idx cache path")
        self.parser.add_argument("--idx2word_path", type=str, default="./cache/idx2word.pickle",
                                 help="idx2word cache path")
        self.parser.add_argument("--vocab_embedding_path", type=str, default="./cache/vocab_embedding.pickle",
                                 help="vocab_embedding cache path")
        self.initialized = True

    def display_save(self, options, results_dir):
        """save config info for future reference, and print"""
        args = vars(options)  # type == dict
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(results_dir, 'opt.json')  # not yaml file indeed
            save_json_pretty(args, option_file_path)

    def parse(self):
        """parse cmd line arguments and do some preprocessing"""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
                            
        if opt.save_name:
            results_dir = opt.results_dir_base + opt.save_name           
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            self.display_save(opt, results_dir)
            opt.results_dir = results_dir

        opt.normalize_v = not opt.no_normalize_v
        opt.with_ts = not opt.no_ts
        opt.input_streams = [] if opt.input_streams is None else opt.input_streams
        opt.vid_feat_flag = True if "imagenet" in opt.input_streams else False
        opt.h5driver = None if opt.no_core_driver else "core"
        
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file")
        self.parser.add_argument("--mode", type=str, default="valid", help="valid/test")


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--input_streams", "vcpt"]
    opt = BaseOptions().parse()

