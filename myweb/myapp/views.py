from __future__ import unicode_literals
from django.shortcuts import render
from myapp import models
from keras.models import load_model as keras_load_model
import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as Fun
# Create your views here.

def index(request):
    return render(request, 'index.html')


def predict_gender(request):
    if request.method == "POST":
        predict_gender_conversation = request.POST.get("predict_gender_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")
        #load_model = keras_load_model("D:/mlwc/iris_model/iris_modeliris_model.h5")

        np.set_printoptions(precision=4)

        unknown = np.array([predict_gender_conversation.split(" ")], dtype=np.float32)
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

        return render(request, 'predict_gender.html', {'predict_gender_result': pred_y})

    return render(request, 'predict_gender.html')

def predict_character(request):
    if request.method == "POST":
        predict_character_conversation = request.POST.get("predict_character_conversation", None)
        password = request.POST.get("password", None)

        print("Using loaded model to predict......")
        #load_model = keras_load_model("D:/mlwc/iris_model/iris_modeliris_model.h5")

        np.set_printoptions(precision=4)

        unknown = np.array([predict_character_conversation.split(" ")], dtype=np.float32)
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

        return render(request, 'predict_character.html', {'predict_character_result': pred_y})

    return render(request, 'predict_character.html')

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