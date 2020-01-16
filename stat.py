import numpy as np
import matplotlib.pyplot as plt
import os

def stat():
    with open('ml_result.txt', 'r', encoding='utf-8') as f:
        res = [[], [], []]
        for line in f.read().split('\n')[1:]:
            if line:
                items = line.split('\t')
                if items[1] == 'genre':
                    for i in range(3):
                        res[i].append(float(items[i-3]))
        print('{:.3f} {:.3f} {:.3f}'.format(np.mean(res[0]), np.mean(res[1]), np.mean(res[2])))

def draw(name, vocab_dims=None, dialog_dims=None, feature_dims=None):
    vocab_dim, dialog_dim, feature_dim = 32, 128, 256
    f1s = []
    labels = []
    for model in ['pretrain', 'vocab']:
        for task in ['IMDB', 'genre', 'gender']:
            file = '{}_{}_result.txt'.format(model, task)
            if not os.path.exists(file):
                continue
            f1 = []
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.read().split('\n')[1:]:
                    if line:
                        items = line.split('\t')
                        if vocab_dims and int(items[2]) in vocab_dims:
                            f1.append(float(items[-1]))
                        if dialog_dims and int(items[3]) in dialog_dims:
                            f1.append(float(items[-1]))
                        if feature_dims and int(items[4]) in feature_dims:
                            f1.append(float(items[-1]))
            f1s.append(f1)
            labels.append('{}_{}'.format(model, task))
    if vocab_dims:
        xs = vocab_dims
        ylabel = 'Vocab dimension'
    if dialog_dims:
        xs = dialog_dims
        ylabel = 'Dialog dimension'
    if feature_dims:
        xs = feature_dims
        ylabel = 'Feature dimension'
    plt.clf()
    for i in range(len(f1s)):
        plt.plot(xs, f1s[i], label=labels[i])
    plt.xlabel('F1 score')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('draw/{}.png'.format(ylabel))

if __name__ == '__main__':
    stat()
    draw('vocab_dim', vocab_dims=[16, 32, 48])
    draw('dialog_dim', dialog_dims=[64, 128, 192])
    draw('feature_dim', feature_dims=[128, 256, 384])