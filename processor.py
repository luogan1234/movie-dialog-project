import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    
    def loss_func(self, outputs, labels):
        if self.config.task == 'genre':
            loss = F.binary_cross_entropy(F.sigmoid(outputs), labels)
        else:
            loss = F.cross_entropy(outputs, labels)
        return loss
    
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
            if self.config.task == 'genre':
                predicts[predicts>0.5] = 1
                predicts[predicts<=0.5] = 0
            else:
                predicts = np.argmax(predicts, 1)
        return predicts, loss
    
    def evaluate(self, data, is_print=False):
        self.model.eval()
        eval_steps = (len(data) - 1) // self.config.batch_size + 1
        labels_all = np.array([])
        predicts_all = np.array([])
        res = 0.0
        for i in range(eval_steps):
            inputs, labels = self.get_batch_data(data[i*self.config.batch_size: (i+1)*self.config.batch_size])
            predicts, loss = self.eval_one_step(inputs, labels)
            res += loss
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predicts)
        self.model.train()
        res /= eval_steps
        labels_all = labels_all.flatten()
        predicts_all = predicts_all.flatten()
        acc = metrics.accuracy_score(labels_all, predicts_all)
        p = metrics.precision_score(labels_all, predicts_all, average='weighted')
        r = metrics.recall_score(labels_all, predicts_all, average='weighted')
        f1 = metrics.f1_score(labels_all, predicts_all, average='weighted')

        metric_res = {'loss/eval': res, 'metrics/acc': acc, 'metrics/precision': p, 
                   'metrics/recall': r, 'metrics/f1-score': f1}
        if is_print:
            print('Eval finished, acc {:.3f}, p {:.3f}, r {:.3f}, f1 {:.3f}'.format(acc, p, r, f1))
        if self.config.task == 'IMDB':
            me = np.mean(np.abs(labels_all-predicts_all))
            metric_res['metrics/mean-error'] = me
            if is_print:
                print('mean error: {:.3f}'.format(me))
        return metric_res
    
    def get_batch_data(self, data):
        inputs, labels = [], []
        for sample in data:
            input, label = sample
            inputs.append(input)
            labels.append(label)
        inputs = self.config.to_torch(np.array(inputs))
        if self.config.task == 'genre':
            labels = np.array(labels, dtype=np.float32)
        else:
            labels = np.array(labels, dtype=np.int64)
        return inputs, labels
    
    def train(self, output):
        if self.config.use_gpu:
            self.model.cuda()
        writer = SummaryWriter('runs/' + self.config.to_str())
        best_para = self.model.state_dict()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train = self.store.train
        train_steps = (len(train) - 1) // self.config.batch_size + 1
        
        min_loss = 1e16
        try:
            for epoch in range(self.config.epochs):
                training_range = tqdm.tqdm(range(train_steps))
                training_range.set_description("Epoch %d, Iter %d | loss: " % (epoch, 0))
                log_res = 0.0
                random.shuffle(train)
                for i in training_range:
                    inputs, labels = self.get_batch_data(train[i*self.config.batch_size: (i+1)*self.config.batch_size])
                    loss = self.train_one_step(inputs, labels)
                    log_res += loss
                    if i > 0 and i % self.config.log_interval == 0:
                        log_res /= self.config.log_interval
                        training_range.set_description("Epoch %d, Iter %d | loss: %.3f" % (epoch, i, log_res))
                        writer.add_scalar('loss/train', log_res, epoch * train_steps + i)
                        log_res = 0.0
                        metric_res = self.evaluate(self.store.eval)
                        for key, value in metric_res.items():
                            writer.add_scalar(key, value, epoch * train_steps + i)
                        if metric_res['loss/eval'] < min_loss:
                            min_loss = metric_res['loss/eval']
                            best_para = self.model.state_dict()
        except KeyboardInterrupt:
            writer.close()
            print('-' * 89)
            print('Exiting from training early')

        self.model.load_state_dict(best_para)
        self.evaluate(self.store.test, True)
