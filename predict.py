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
from bert_serving.client import BertClient


def name_to_model(name, config):
    if name == 'pretrain':
        return PretrainModel(config)
    if name == 'vocab':
        return VocabModel(config)
    raise NotImplementedError

def evaluate(model, inputs, task, max_dialog_words):
    model.eval()

    inputs_padding = []
    for conversation in inputs:
        if len(conversation) <= max_dialog_words:
            inputs_padding.append(conversation + [0] * (max_dialog_words - len(conversation)))
        else:
            inputs_padding.append(conversation[:max_dialog_words])
    if len(inputs_padding) < 7:
        for _ in range(7 - len(inputs_padding)):
            inputs_padding.append([0] * max_dialog_words)
    inputs_padding = np.array(inputs_padding)
    inputs = torch.from_numpy(inputs_padding).cuda().unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        predicts = outputs.data.cpu().numpy()[0]
        print(predicts)
        if task == 'genre':
            predicts = [(i, predicts[i]) for i in range(len(predicts))]
            predicts.sort(key=lambda x:x[1], reverse=True)
        elif task == 'IMDB':
            predicts = [(i, predicts[i]) for i in range(len(predicts))]
            predicts.sort(key=lambda x:x[1], reverse=True)
    print(predicts)
    return predicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie dialog')
    parser.add_argument('-model', type=str, required=True, choices=['pretrain', 'vocab'], help='pretrain | vocab')
    parser.add_argument('-task', type=str, required=True, choices=['IMDB', 'genre', 'gender'], help='IMDB | genre | gender')
    parser.add_argument('-use_cpu', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-max_dialog_words', type=int, default=64)
    parser.add_argument('-vocab_dim', type=int, default=32)
    parser.add_argument('-dialog_dim', type=int, default=128)
    parser.add_argument('-feature_dim', type=int, default=256)
    parser.add_argument('-log_interval', type=int, default=10)

    args = parser.parse_args()

    store = DataHandler(args.model, args.task, args.max_dialog_words)
    config = Config(store, args.use_cpu, args.epochs, args.vocab_dim, args.dialog_dim, args.feature_dim, args.log_interval)
    model = name_to_model(args.model, config)
    save_path = os.path.join(config.save_dir, '{}_{}_{}_{}_{}.ckpt'.format(args.model, args.task, args.vocab_dim, args.dialog_dim, args.feature_dim))
    model.load_state_dict(torch.load(save_path))
    model.cuda()

    vocab_map = np.load('./dataset/vocab_map.npy', allow_pickle=True)[()]
    print(list(vocab_map.items())[:5])

    bc = BertClient()
    movie = ['First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it First do it', 
             'then do it right', 'then do it better']
    vecs, tokens = bc.encode(movie, show_tokens=True)
    print(tokens)
    token_index = []
    for dialog in tokens:
        token_index.append([])
        for token in dialog[1:-1]:
            if token in vocab_map:
                token_index[-1].append(vocab_map[token])
    print(token_index)
    evaluate(model, token_index, args.task, args.max_dialog_words)