# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 19:50:55 2023

@author: jugwu
"""

import torch
import math
from tqdm.notebook import tqdm
import numpy as np
from IPython.display import clear_output
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#__________________________________________________
def flatten(feature_map):
    flattened_vector = torch.flatten(feature_map)
    return flattened_vector
#__________________________________________________
def batch_input(input_batch):
    if len(input_batch.shape) > 2:
        _input = []
        
        for feature_map in input_batch:
            _input.append(flatten(feature_map))
        
        return torch.stack(_input)          
    else:
        return flatten(input_batch)
#__________________________________________________
def stack(feature_maps):
    flattened_vectors = []
    
    for feature_map in feature_maps:
        flattened_vectors.append(flatten(feature_map))
    
    output_vector = torch.stack(flattened_vectors)
    return flatten(output_vector)
#__________________________________________________
class MSELoss(object):
    def __init__(self, padding:bool=False):
        self.padding = padding
        return
    
    def calculate(self, predictions, labels):

        self.labels = labels
        self.predictions = predictions

        if self.padding:
            self.bool_tensor = torch.where(torch.argmax(labels, dim=-1) == 1, 0, 1).float()
            self.bool_tensor = self.bool_tensor.unsqueeze(-1).expand(-1,-1,self.labels.size()[-1])
            _sum = torch.mean(((predictions - labels) ** 2)* self.bool_tensor)
        else:
            _sum = torch.mean(((predictions - labels) ** 2))

        loss = _sum / (2 * self.labels.size()[0])

        return loss
    
    def back(self):
        
        self.gradient = 2 * (self.predictions - self.labels) / self.labels.size()[0]

        if self.padding:
            self.gradient = self.gradient * self.bool_tensor
        
        return self.gradient
#__________________________________________________
class CrossEntropyLoss(object):
    def __init__(self, padding:bool=False):
        self.padding = padding
        return
    
    def calculate(self, predictions, labels):

        self.labels = labels
        self.predictions = predictions

        if self.padding:
            self.bool_tensor = torch.where(torch.argmax(labels, dim=-1) == 1, 0, 1).float()
            self.bool_tensor = self.bool_tensor.unsqueeze(-1).expand(-1,-1,self.labels.size()[-1])
            loss = - torch.mean((labels * torch.log(predictions + 1e-15))* self.bool_tensor)
        else:
            loss = - torch.mean((labels * torch.log(predictions + 1e-15)))

        return loss
    
    def back(self):
    
        self.gradient = -(self.labels/self.predictions + 1e-15)/len(self.labels)
        
        if self.padding:
            self.gradient = self.gradient * self.bool_tensor

        return self.gradient
#__________________________________________________
class BinaryCrossEntropyLoss(object):
    def __init__(self, padding:bool=False):
        self.padding = padding
        return
    
    def calculate(self, predictions, labels):

        self.labels = labels
        self.predictions = predictions
        
        if self.padding:
            self.bool_tensor = torch.where(torch.argmax(labels, dim=-1) == 1, 0, 1).float()
            self.bool_tensor = self.bool_tensor.unsqueeze(-1).expand(-1,-1,self.labels.size()[-1])
            loss = - torch.mean((labels * torch.log(predictions + 1e-15) + (1-labels)*torch.log(1-(predictions + 1e-15)))* self.bool_tensor)
        else:
            loss = - torch.mean((labels * torch.log(predictions + 1e-15) + (1-labels)*torch.log(1-(predictions + 1e-15))))
        
        return loss
    
    def back(self):
        
        self.gradient = -((self.labels/(self.predictions + 1e-15)) - (1-self.labels)/(1-self.predictions + 1e-15))/len(self.labels)
        if self.padding:
            self.gradient = self.gradient * self.bool_tensor

        return self.gradient
#__________________________________________________
class TrainManager(object):
    def __init__(self, train, test, val = None):
        self.train_loader = train
        self.test_loader = test
        self.val_loader = val
        self.history = {
            'train_loss' : [],
            'validation_loss' : [],
            'train_accuracy' : [],
            'validation_accuracy' : [],
            'test_accuracy' : []
        }
        return
    
    def one_hot(self, number):
        result = torch.zeros(10, dtype=torch.float)
        result[number] = 1.0
        return result

    def train(self, _model, loss_func = MSELoss, epochs:int=1, lr:float=0.001, optim_b1:float=0.9, optim_b2:float=0.999, num_batches:int=1, optimiser='adam', decay:bool=False, loss_plot:bool=False):
       
        count = 0

        if decay:
            decay = lr/epochs

        for epoch in tqdm(range(epochs)):

            total = 0
            correct = 0
            batch_count = 0

            for train_batch in self.train_loader:

                if decay:
                    lr *= (1/(1 + decay * (epoch * num_batches + total)))

                count+=1
                total+=1

                batch_count += 1
                if batch_count > num_batches:
                    break
                    
                _input = train_batch[0]
                _label = torch.stack(list(map(self.one_hot, train_batch[1])))

                result = _model.forward(_input)
                _loss = loss_func.calculate(result, _label)
                _model.backward(loss_func, lr=lr, optim=optimiser, optim_b1=optim_b1, optim_b2=optim_b2)
                self.history['train_loss'].append(_loss)
                
                corr_ten = (torch.argmax(result, dim=1) == torch.argmax(_label, dim=1)).float()
                correct += torch.mean(corr_ten) 
                self.history['train_accuracy'].append((correct/total) * 100)

                if loss_plot:
                    self.plot_loss_accuracy()
                print(f"Epoch: {epoch}, it: {count}/{num_batches * epochs} [train_loss: {self.history['train_loss'][-1]:.5f}], [train_accuracy: {self.history['train_accuracy'][-1]:.1f}]")
        
        self.trained_model = _model

        return
    
    def test(self, num_batches:int=100, num_examples:int=1):

        total = 0
        correct = 0
        batch_count = 0

        for test_batch in tqdm(self.test_loader):
            total+=1

            batch_count += 1
            if batch_count > num_batches:
                break
            
            _input = test_batch[0]
            _label = torch.stack(list(map(self.one_hot, test_batch[1])))

            result = self.trained_model.forward(_input)    
            corr_ten = (torch.argmax(result, dim=1) == torch.argmax(_label, dim=1)).float()
            correct += torch.mean(corr_ten) 

            self.history['test_accuracy'].append((correct/total) * 100)

        result = self.trained_model.forward_show(_input, _label, num_examples)   
        print(f"[test_accuracy: {sum(self.history['test_accuracy'])/len(self.history['test_accuracy'])}]")

        return
    
    def plot_loss_accuracy(self):
        
        clear_output(wait=True)
        fig, ax = plt.subplots(2)
        
        ax[0].plot(self.history['train_loss'], label="train_loss")
        #ax[0].plot(self.history['val_loss'], label="validation_loss")
        #ax[0].set_ylim(0, 0.2)
        #ax[0].set_xlim(0, self.data_size)
        ax[0].legend()
        
        ax[1].plot(self.history['train_accuracy'], label="train_accuracy")
        #ax[1].plot(self.history['valid_accuracy'], label="validation_accuracy")
        ax[1].set_ylim(0, 100)
        #ax[1].set_xlim(0, self.data_size)
        ax[1].legend()
        plt.show()
        
        return