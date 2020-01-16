from __future__ import unicode_literals
from django.shortcuts import render
from myapp import models
import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as F
import sys
import os

from bert_serving.client import BertClient
import myapp.apps as apps

# Create your views here.

def index(request):
    return render(request, 'index.html')

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
        if task == 'IMDB':
            outputs = F.softmax(outputs, -1)
        elif task == 'genre':
            outputs = torch.sigmoid(outputs)
        elif task == 'gender':
            outputs = F.softmax(outputs, -1)
        predicts = outputs.data.cpu().numpy()[0]
        predicts = [(i, predicts[i]) for i in range(len(predicts))]
        predicts.sort(key=lambda x:x[1], reverse=True)
    print(predicts)
    return predicts

def load_model(model, task):
    return apps.preload_models[task]

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
        token_index = []
        for dialog in tokens:
            token_index.append([])
            for token in dialog[1:-1]:
                if token in vocab_map:
                    token_index[-1].append(vocab_map[token])
        predicts = evaluate(model, token_index, 'gender', 64)
        results = []
        for index, pred in predicts:
            results.append(['female' if index == 0 else 'male', pred])

        return render(request, 'predict_gender.html', {'predict_gender_result': results})

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
        token_index = []
        for dialog in tokens:
            token_index.append([])
            for token in dialog[1:-1]:
                if token in vocab_map:
                    token_index[-1].append(vocab_map[token])
        predicts = evaluate(model, token_index, 'genre', 64)[:3]
        results = []
        for index, pred in predicts:
            results.append([apps.genre_rmap[index], pred])

        return render(request, 'predict_genre.html', {'predict_genre_result': results})

    return render(request, 'predict_genre.html')

def predict_rating(request):
    if request.method == "POST":
        predict_rating_conversation = request.POST.get("predict_rating_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")

        model = load_model('vocab', 'IMDB')

        vocab_map = np.load('/home/yukuo/movie-dialog-project/dataset/vocab_map.npy', allow_pickle=True)[()]

        bc = BertClient()
        data = predict_rating_conversation.split('\n')
        print(data)
        vecs, tokens = bc.encode(data, show_tokens=True)
        token_index = []
        for dialog in tokens:
            token_index.append([])
            for token in dialog[1:-1]:
                if token in vocab_map:
                    token_index[-1].append(vocab_map[token])
        predicts = evaluate(model, token_index, 'IMDB', 64)[:3]

        return render(request, 'predict_rating.html', {'predict_rating_result': predicts})

    return render(request, 'predict_rating.html')

def list(request):
    people_list = models.message.objects.all()
    return render(request, 'show.html', {"people_list":people_list})

def test(request):
    return render(request, 'test.html')