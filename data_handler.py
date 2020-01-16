import numpy as np
import os
import json
import tqdm
import random
import pickle

class DataHandler:
    def __init__(self, model, task, max_dialog_words):
        self.model_name = model
        self.task = task
        self.max_dialog_words = max_dialog_words
        if self.task in ['IMDB', 'genre']:
            self.max_length = 24
        else:
            self.max_length = 16
        self.vocabs = np.load('dataset/vocab.npy', allow_pickle=True)
    
    def padding(self, conversation):
        mat = []
        if self.model_name == 'vocab':
            for c in conversation:
                d = c[:self.max_dialog_words]
                mat.append(d+[0]*(self.max_dialog_words-len(d)))
            mat = np.array(mat)
            mat = mat[:self.max_length]
            mat = np.concatenate([mat, np.zeros((self.max_length-mat.shape[0], mat.shape[1]))], 0)
        else:
            d = conversation
            mat = np.array(d+[0]*(self.max_length-len(d)))
        mat = np.array(mat, dtype=np.int64)
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
            if self.task in ['IMDB', 'genre']:
                if self.model_name == 'pretrain':
                    movies = np.load('dataset/movie_pretrain.npz', allow_pickle=True)
                else:
                    movies = np.load('dataset/movie_vocab.npz', allow_pickle=True)
                movie_num = len(movies['data'])
            if self.task in ['gender']:
                if self.model_name == 'pretrain':
                    characters = np.load('dataset/character_pretrain.npz', allow_pickle=True)
                else:
                    characters = np.load('dataset/character_vocab.npz', allow_pickle=True)
                character_num = len(characters['data'])
            groups = []
            if self.task in ['IMDB', 'genre']:
                for i in tqdm.tqdm(range(movie_num)):
                    conversations = movies['data'][i]
                    rating = movies['ratings'][i]
                    genre = self.one_hot(movies['genres'][i], 24)
                    n = len(conversations)
                    for i in range(0, n-self.max_length, self.max_length):
                        mat = self.padding(conversations[i:i+self.max_length])
                        if self.task == 'IMDB':
                            groups.append([mat, rating])
                        else:
                            groups.append([mat, genre])
            if self.task == 'gender':
                for i in tqdm.tqdm(range(character_num)):
                    conversations = characters['data'][i]
                    gender = characters['genders'][i]
                    n = len(conversations)
                    for i in range(0, n-self.max_length, self.max_length):
                        mat = self.padding(conversations[i:i+self.max_length])
                        groups.append([mat, gender])
        with open(store_path, 'wb') as f:
            pickle.dump(groups, f)
        random.shuffle(groups)
        d1, d2 = int(len(groups)*0.8), int(len(groups)*0.9)
        self.train = groups[:d1]
        self.eval = groups[d1:d2]
        self.test = groups[d2:]
        print('prepare data done.')