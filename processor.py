import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import os
import json
import numpy as np
import tqdm
import random

class Processor(object):
    def __init__(self, model, store, config):
        self.model = model
        self.store = store
        self.config = config
        if config.task == 'classification':
            self.loss_func = F.binary_cross_entropy
        else:
            self.loss_func = F.l1_loss
    
    def train_one_step(self, inputs, labels):
        labels = self.config.to_torch(labels)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval_one_step(self, inputs, labels):
        labels = self.config.to_torch(labels)
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            predicts = outputs.data.cpu().numpy()
            if self.config.task == 'classification':
                predicts[predicts>0.5] = 1
                predicts[predicts<=0.5] = 0
            else:
                predicts = np.round(predicts*10)/10
        return predicts, loss
    
    def get_batch_data(self, data):
        inputs, labels = [], []
        for sample in data:
            input, label = sample
            inputs.append(input)
            labels.append(label)
        inputs = self.config.to_torch(np.array(inputs))
        if config.task == 'classification':
            labels = np.array(labels, dtype=np.int64)
        else:
            labels = np.array(labels, dtype=np.float32)
        return inputs, labels
    
    def run(self, output):
        if self.config.use_gpu:
            self.model.cuda()
        best_para = self.model.state_dict()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train, eval = self.store.prepare_data()
        train_steps = (len(train) - 1) // self.config.batch_size + 1
        eval_steps = (len(eval) - 1) // self.config.batch_size + 1
        training_range = tqdm.tqdm(range(self.config.epochs))
        training_range.set_description("Epoch %d | loss: %.3f" % (0, 0))
        min_loss = 1e16
        for epoch in training_range:
            res = 0.0
            random.shuffle(train)
            for i in range(train_steps):
                inputs, labels = self.get_batch_data(train[i*self.config.batch_size: (i+1)*self.config.batch_size])
                loss = self.train_one_step(inputs, labels)
                res += loss
            training_range.set_description("Epoch %d | loss: %.3f" % (epoch, res))
            if res < min_loss:
                min_loss = res
                best_para = self.model.state_dict()
        self.model.load_state_dict(best_para)
        print('Train finished, min_loss {:.3f}'.format(min_loss))
        self.model.eval()
        labels_all = np.array([])
        predicts_all = np.array([])
        for i in range(eval_steps):
            inputs, labels = self.get_batch_data(eval[i*self.config.batch_size: (i+1)*self.config.batch_size])
            predicts, loss = self.eval_one_step(inputs, labels)
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predicts)
        labels_all = labels_all.flatten()
        predicts_all = predicts_all.flatten()
        if self.config.task == 'classification':
            acc = metrics.accuracy_score(labels_all, predicts_all)
            p = metrics.precision_score(labels_all, predicts_all, average='weighted')
            r = metrics.recall_score(labels_all, predicts_all, average='weighted')
            f1 = metrics.f1_score(labels_all, predicts_all, average='weighted')
            print('Eval finished, acc {:.3f}, p {:.3f}, r {:.3f}, f1 {:.3f}'.format(acc, p, r, f1))
        else:
            error = np.mean(np.abs(labels_all-predicts_all))
            print('Eval finished, mean error {:.3f}'.format(error))