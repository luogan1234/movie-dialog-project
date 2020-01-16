import numpy as np
from sklearn import svm, metrics
import argparse
import os
import random

def prepare_data(model, task):
    data = np.load('tmp/store_{}_{}.pkl'.format(model, task), allow_pickle=True)
    random.shuffle(data)
    d = int(len(data)*0.9)
    train, test = data[:d], data[d:]
    if model == 'pretrain':
        dialog_embedding = np.load('dataset/embed_pca.npz', allow_pickle=True)['emb']
        for t in train:
            t[0] = np.mean([dialog_embedding[index][:128] for index in t[0]], 0)
        for t in test:
            t[0] = np.mean([dialog_embedding[index][:128] for index in t[0]], 0)
    if model == 'vocab':
        vocabs = np.load('dataset/vocab.npy', allow_pickle=True)
        for t in train:
            words = set(t[0].flatten())
            t[0] = np.array([float(i in words) for i in range(len(vocabs))], dtype=np.float32)
        for t in test:
            words = set(t[0].flatten())
            t[0] = np.array([float(i in words) for i in range(len(vocabs))], dtype=np.float32)
    return train, test

def evaluate(model, task, data, genre_id, regression):
    train_X = [item[0] for item in data[0]]
    train_Y = [item[1] for item in data[0]]
    test_X = [item[0] for item in data[1]]
    test_Y = [item[1] for item in data[1]]
    if task == 'IMDB' and not regression:
        train_Y = [int(y) for y in train_Y]
        test_Y = [int(y) for y in test_Y]
    if task == 'genre':
        train_Y = [y[genre_id] for y in train_Y]
        test_Y = [y[genre_id] for y in test_Y]
    if task == 'IMDB' and regression:
        ml = svm.SVR()
        ml.fit(train_X, train_Y)
        pred_Y = ml.predict(test_X)
        mean_error = np.mean(np.abs(test_Y-pred_Y))
        print('mean_error: {:.3f}'.format(mean_error))
    else:
        ml = svm.SVC()
        ml.fit(train_X, train_Y)
        pred_Y = ml.predict(test_X)
        acc = metrics.accuracy_score(test_Y, pred_Y)
        p = metrics.precision_score(test_Y, pred_Y, average='weighted')
        r = metrics.recall_score(test_Y, pred_Y, average='weighted')
        f1 = metrics.f1_score(test_Y, pred_Y, average='weighted')
        print('acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}'.format(acc, p, r, f1))
        output = 'ml_result.txt'
        if not os.path.exists(output):
            with open(output, 'w', encoding='utf-8') as f:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('model', 'task', 'genre_id', 'accuracy', 'precision', 'recall', 'f1-score'))
        with open(output, 'a', encoding='utf-8') as f:
            f.write('{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(model, task, genre_id, acc, p, r, f1))

def main():
    parser = argparse.ArgumentParser(description='Movie dialog')
    parser.add_argument('-model', type=str, required=True, choices=['pretrain', 'vocab'], help='pretrain | vocab')
    parser.add_argument('-task', type=str, required=True, choices=['IMDB', 'genre', 'gender'], help='IMDB | genre | gender')
    parser.add_argument('-genre_id', type=int, default=0)
    parser.add_argument('-regression', action='store_true')
    args = parser.parse_args()
    data = prepare_data(args.model, args.task)
    evaluate(args.model, args.task, data, args.genre_id, args.regression)

if __name__ == '__main__':
    main()