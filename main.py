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
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-max_dialog_words', type=int, default=128)
    parser.add_argument('-max_sequence_length', type=int, default=64)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-save_dir', type=str, default='result/')
    args = parser.parse_args()
    if not os.path.exists('result/'):
        os.mkdir('result/')
    store = DataHandler(args.model, args.task, args.max_dialog_words, args.max_sequence_length)
    config = Config(store, args.use_cpu, args.epochs)
    model = name_to_model(args.model, config)
    processor = Processor(model, store, config)
    processor.train(args.output)
    if args.save_dir:
        save_path = os.path.join(config.save_dir, '{}_{}.ckpt'.format(args.model, args.task))
        print("Model save at {}".format(config.save_dir))
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    train()