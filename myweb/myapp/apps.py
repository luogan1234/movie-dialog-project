import sys
from django.apps import AppConfig

preload_models = {}
genre_rmap = {}

def load_model(model, task):
    import numpy as np
    import torch
    import tensorflow as tf
    import torch.nn.functional as F
    import os
    sys.path.append("..")
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("..") for name in dirs])
    from data_handler import DataHandler
    from processor import Processor
    from config import Config
    from models.pretrain_model import PretrainModel
    from models.vocab_model import VocabModel

    def name_to_model(name, config):
        if name == 'pretrain':
            return PretrainModel(config)
        if name == 'vocab':
            return VocabModel(config)
        raise NotImplementedError

    use_cpu = False
    epochs = 20
    max_dialog_words = 64
    vocab_dim = 32
    dialog_dim = 128
    feature_dim = 256
    log_interval = 10

    genre_map = np.load('/home/yukuo/movie-dialog-project/genre_map.npy', allow_pickle=True)[()]
    global genre_rmap
    for key, value in genre_map.items():
        genre_rmap[value] = key

    store = DataHandler(model, task, max_dialog_words)
    config = Config(store, use_cpu, epochs, vocab_dim, dialog_dim, feature_dim, log_interval)
    save_path = os.path.join('/home/yukuo/movie-dialog-project/result', '{}_{}_{}_{}_{}.ckpt'.format(model, task, vocab_dim, dialog_dim, feature_dim))
    model = name_to_model(model, config)
    model.load_state_dict(torch.load(save_path))
    model.cuda()
    return model

class MyappConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        if 'runserver' not in sys.argv:
            return True
        # you must import your modules here 
        # to avoid AppRegistryNotReady exception 

        # startup code here
        global preload_models
        preload_models['gender'] = load_model('vocab', 'gender')
        preload_models['genre'] = load_model('vocab', 'genre')
        preload_models['IMDB'] = load_model('vocab', 'IMDB')
