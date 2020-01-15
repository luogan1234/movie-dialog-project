import numpy as np
import os
import json
import tqdm
import random
import pickle

class DataHandler:
    def __init__(self, model, task, max_dialog_words, max_length):
        self.model_name = model
        self.task = task
        self.max_dialog_words = max_dialog_words
        self.max_length = max_length
        if self.model_name == 'pretrain':
            self.movies = np.load('dataset/movie_pretrain.npz', allow_pickle=True)
        else:
            self.movies = np.load('dataset/movie_vocab.npz', allow_pickle=True)
        self.movie_num = len(self.movies['data'])
        self.vocabs = np.load('dataset/vocab.npy', allow_pickle=True)
    
    def padding(self, conversation):
        mat = []
        if self.model_name == 'vocab':
            for c in conversation:
                d = c[:self.max_dialog_words]
                mat.append(d+[0]*(self.max_dialog_words-len(d)))
            mat = np.array(mat, dtype=np.int64)
            mat = mat[:self.max_length]
            mat = np.concatenate([mat, np.zeros((self.max_length-mat.shape[0], mat.shape[1]), dtype=np.int64)], 0)
        else:
            mat = np.array(conversation, dtype=np.float32)
            mat = mat[:self.max_length]
            mat = np.concatenate([mat, np.zeros((self.max_length-mat.shape[0], mat.shape[1]), dtype=np.float32)], 0)
        return mat
    
    def one_hot(self, classes, num):
        vec = [float(i in classes) for i in range(num)]
        return vec
    
    def prepare_data(self):
        store_path = 'tmp/store_{}_{}.pkl'.format(self.model_name, self.task)
        if os.path.exists(store_path):
            with open(store_path, 'rb') as f:
                groups = pickle.load(f)
        else:
            groups = []
            if self.task in ['IMDB', 'genre']:
                for i in tqdm.tqdm(range(self.movie_num)):
                    conversations = self.movies['data'][i]
                    rating = self.movies['ratings'][i]
                    genre = self.one_hot(self.movies['genres'][i], 24)
                    n = len(conversations)
                    for i in range(0, n-self.max_length, self.max_length):
                        mat = self.padding(conversations[i:i+self.max_length])
                        if self.task == 'IMDB':
                            groups.append([mat, rating])
                        else:
                            groups.append([mat, genre])
            if self.task == 'gender':
                pass
        with open(store_path, 'wb') as f:
            pickle.dump(groups, f)
        random.shuffle(groups)
        d = int(len(groups)*0.9)
        self.train = groups[:d]
        self.eval = groups[d:]
        print('preare data done.')