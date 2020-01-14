import numpy as np
import os
import json
import tqdm
import random

class DataHandler:
    def __init__(self, model, task, max_dialog_words, max_sequence_length):
        self.model_name = model
        self.task = task
        self.max_dialog_words = max_dialog_words
        self.max_sequence_length = max_sequence_length
        if self.model_name == 'pretrain':
            self.movies = np.load('movie_pretrain.npz', allow_pickle=True)
            self.movie_num = len(self.movies['data'])
        else:
            pass
    
    def padding(self, conversation):
        mat = []
        if self.model_name == 'vocab':
            for c in conversation:
                mat.append(c+[0]*self.max_dialog_words)
            for i in range(self.max_sequence_length-len(mat)):
                mat.append([0]*self.max_dialog_words)
            mat = np.array(mat, dtype=np.int64)
        else:
            for i in range(self.max_sequence_length-mat(vec)):
                mat.append([0]*768)
            mat = np.array(mat, dtype=np.float32)
        return mat
    
    def prepare_data():
        groups = []
        if self.task in ['IMDB', 'genre']:
            for i in range(self.movie_num):
                conversations = self.movies['data'][i]
                rating = self.movies['ratings'][i]
                genre = self.movies['genres'][i]
                for conversation in conversations:
                    mat = self.padding(conversation)
                    if self.task == 'IMDB':
                        groups.append([mat, rating])
                    else:
                        groups.append([mat, genre])
        if self.task == 'gender':
            pass
        random.shuffle(groups)
        d = int(len(groups)*0.9)
        return groups[:d], groups[d:]