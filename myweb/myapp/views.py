from __future__ import unicode_literals
from django.shortcuts import render
from myapp import models
import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as Fun
import sys
import os

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("..") for name in dirs])

from data_handler import DataHandler
from processor import Processor
from config import Config
from models.pretrain_model import PretrainModel
from models.vocab_model import VocabModel
from bert_serving.client import BertClient

# Create your views here.

def index(request):
    return render(request, 'index.html')

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
        predicts = [(i, predicts[i]) for i in range(len(predicts))]
        predicts.sort(key=lambda x:x[1], reverse=True)
    print(predicts)
    return predicts

def load_model(model, task):
    use_cpu = False
    epochs = 20
    max_dialog_words = 64
    vocab_dim = 32
    dialog_dim = 128
    feature_dim = 256
    log_interval = 10

    store = DataHandler(model, task, max_dialog_words)
    config = Config(store, use_cpu, epochs, vocab_dim, dialog_dim, feature_dim, log_interval)
    save_path = os.path.join('/home/yukuo/movie-dialog-project/result', '{}_{}_{}_{}_{}.ckpt'.format(model, task, vocab_dim, dialog_dim, feature_dim))
    model = name_to_model(model, config)
    model.load_state_dict(torch.load(save_path))
    model.cuda()
    return model

def predict_gender(request):
    if request.method == "POST":
        predict_gender_conversation = request.POST.get("predict_gender_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")

        model = load_model('vocab', 'gender')

        vocab_map = np.load('/home/yukuo/movie-dialog-project/dataset/vocab_map.npy', allow_pickle=True)[()]

        bc = BertClient()
        data = predict_gender_conversation.split('\n')
        print(data)
        vecs, tokens = bc.encode(data, show_tokens=True)
        print(tokens)
        token_index = []
        for dialog in tokens:
            token_index.append([])
            for token in dialog[1:-1]:
                if token in vocab_map:
                    token_index[-1].append(vocab_map[token])
        print(token_index)
        evaluate(model, token_index, 'gender', 64)

        return render(request, 'predict_gender.html', {'predict_gender_result': pred_y})

    return render(request, 'predict_gender.html')

def predict_genre(request):
    if request.method == "POST":
        predict_genre_conversation = request.POST.get("predict_genre_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")

        model = load_model('vocab', 'genre')

        vocab_map = np.load('/home/yukuo/movie-dialog-project/dataset/vocab_map.npy', allow_pickle=True)[()]

        bc = BertClient()
        data = predict_genre_conversation.split('\n')
        print(data)
        vecs, tokens = bc.encode(data, show_tokens=True)
        print(tokens)
        token_index = []
        for dialog in tokens:
            token_index.append([])
            for token in dialog[1:-1]:
                if token in vocab_map:
                    token_index[-1].append(vocab_map[token])
        print(token_index)
        predicts = evaluate(model, token_index, 'genre', 64)[:5]

        return render(request, 'predict_genre.html', {'predict_genre_result': predicts})

    return render(request, 'predict_genre.html')

def predict_rating(request):
    if request.method == "POST":
        predict_rating_conversation = request.POST.get("predict_rating_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")
        #load_model = keras_load_model("D:/mlwc/iris_model/iris_modeliris_model.h5")

        np.set_printoptions(precision=4)

        unknown = np.array([predict_rating_conversation.split(" ")], dtype=np.float32)
        unknown = torch.Tensor(unknown)
        print(type(unknown))
        net = torch.load('D:/mlwc/iris_model/net.pkl')
        predicted = Fun.softmax(net(unknown))
        print("\nfinally.......Predicted softmax vector is: ")
        print(predicted)
        prediction = torch.max(predicted, 1)[1]  # 1返回index  0返回原值
        pred_y = prediction.data.numpy()
        print("predicted_lable=", pred_y)

        #twz = models.message.objects.create(username=username, password=password, predict_result=predicted)
        #objects.create 往数据表中插入内容的方法
        #twz.save()

        return render(request, 'predict_rating.html', {'predict_rating_result': pred_y})

    return render(request, 'predict_rating.html')

def list(request):
    people_list = models.message.objects.all()
    return render(request, 'show.html', {"people_list":people_list})

def test(request):
    return render(request, 'test.html')