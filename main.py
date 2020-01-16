import argparse
from data_handler import DataHandler
from processor import Processor
from config import Config
import pickle
import os
import torch
import random
import numpy as np
from models.pretrain_model import PretrainModel
from models.vocab_model import VocabModel

def name_to_model(name, config):
    if name == 'pretrain':
        return PretrainModel(config)
    if name == 'vocab':
        return VocabModel(config)
    raise NotImplementedError

def train():
    parser = argparse.ArgumentParser(description='Movie dialog')
    parser.add_argument('-model', type=str, required=True, choices=['pretrain', 'vocab'], help='pretrain | vocab')
    parser.add_argument('-task', type=str, required=True, choices=['IMDB', 'genre', 'gender'], help='IMDB | genre | gender')
    parser.add_argument('-use_cpu', action='store_true')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-max_dialog_words', type=int, default=64)
    parser.add_argument('-vocab_dim', type=int, default=32)
    parser.add_argument('-dialog_dim', type=int, default=128)
    parser.add_argument('-feature_dim', type=int, default=256)
    parser.add_argument('-log_interval', type=int, default=10)
    args = parser.parse_args()
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    store = DataHandler(args.model, args.task, args.max_dialog_words)
    store.prepare_data()
    config = Config(store, args.use_cpu, args.epochs, args.vocab_dim, args.dialog_dim, args.feature_dim, args.log_interval)
    model = name_to_model(args.model, config)
    processor = Processor(model, store, config)
    processor.train()
    save_path = os.path.join(config.save_dir, '{}_{}_{}_{}_{}.ckpt'.format(args.model, args.task, args.vocab_dim, args.dialog_dim, args.feature_dim))
    print("Model save at {}".format(config.save_dir))
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    train()