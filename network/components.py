# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 19:50:55 2023

@author: jugwu
"""
import torch
import math
import matplotlib.pyplot as plt

#__________________________________________________
#MODEL PARENT CLASS
#__________________________________________________
class Model(object):
    def __init__(self):
        self.network = []

    def forward(self, _input):

        _output = _input
        for layer in self.network:
            _output = layer.forward(_output)

        return _output
    
    def forward_show(self, _input, _labels, num:int=1):

        _output = _input
        for i, layer in enumerate(self.network):
            _output = layer.forward(_output)
            if len(_output.size()) > 3:
                print(f"LAYER {i}: {_output.size()}")
                plt.figure()
                for j, example in enumerate(_output):
                    if j >= num:
                        break
                    print(f"BATCH: {j}")
                    for feature_map in example:
                        plt.imshow(feature_map, cmap='gray')
                        plt.show()

        for i, (distribution, label) in enumerate(zip(_output, _labels)):
            if i >= num:
                break
            plt.figure()
            plt.bar(torch.range(1, label.size()[0])-1, distribution)
            plt.show()
            print(f"LABEL: {torch.argmax(label)}")
        
        return _output
    
    def backward(self, _loss_function, lr:float=0.001, optim=None,optim_lr:float=0.9, optim_b1:float=0.9, optim_b2:float=0.999):

        _parameters = {
            'lr' : lr,
            'optimiser' : optim,
            'optimiser_lr' : optim_lr,
            'optimiser_b1' : optim_b1,
            'optimiser_b2' : optim_b2
        }

        _gradient = _loss_function.back()
        
        for layer in self.network[::-1]:
            _gradient = layer.back(_gradient, _parameters)

        return
#__________________________________________________
#NORMILISATION
#__________________________________________________
#REGULARISATION
#__________________________________________________
def batch_norm(input_batch):
    _std = torch.zeros_like(input_batch, dtype=torch.float) + torch.std(input_batch)#, dim=0)
    _mean = torch.zeros_like(input_batch, dtype=torch.float) + torch.mean(input_batch)#, dim=0)
    return (input_batch - _mean)/ (_std + 1e-15)
#__________________________________________________
def mini_batch_norm(input_batch):
    _std = torch.std(input_batch, dim = (1,2))
    _std = _std.unsqueeze(-1).unsqueeze(-1).expand(-1,input_batch.size()[-2],input_batch.size()[-1])
    _mean = torch.mean(input_batch, dim = (1,2))
    _mean = _mean.unsqueeze(-1).unsqueeze(-1).expand(-1,input_batch.size()[-2],input_batch.size()[-1])
    return (input_batch - _mean)/ (_std + 1e-15)
#__________________________________________________
def layer_norm_old(input_batch):

    _mean = torch.mean(input_batch, dim = -1)
    _mean = _mean.unsqueeze(-1).expand(-1,-1,input_batch.size()[-1])
    _std = torch.std(input_batch, dim = -1)
    _std = _std.unsqueeze(-1).expand(-1,-1,input_batch.size()[-1])

    return (input_batch - _mean)/ (_std + 1e-15)
#__________________________________________________
def layer_norm(input_batch):
    # inputs : 30 x 200 x 512
    dims = [-(i + 1) for i in range(len(input_batch.size()))] # [-1]
    mean = input_batch.mean(dim=dims, keepdim=True) #30 x 200 x 1
    var = ((input_batch - mean) ** 2).mean(dim=dims, keepdim=True) # 30 x 200 x 512
    std = (var + 1e-15).sqrt() # 30 x 200 x 512
    return (input_batch - mean) / std # 30 x 200 x 512
#__________________________________________________
#LAYERS
class Dropout(object):
    def __init__(self, drop_prob:float=0.1):
        self.probability = drop_prob 
        return
    
    def forward(self, input_matrix):
        self.mask = (torch.rand_like(input_matrix) > self.probability).float() * (1/(1-self.probability))
        return input_matrix * self.mask
    
    def back(self, gradient, param):
        return gradient * self.mask
#__________________________________________________
class FullyConnected(object):
    def __init__(self, input_size, output_size, norm=None):
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = (torch.rand((input_size, output_size), dtype=torch.float) -0.5)/10
        self.bias = torch.zeros(output_size, dtype=torch.float)
        
        self.delta_w = 0
        self.delta_x = 0
        
        self.optim_available = True
        self.optimiser = Gradient_Descent()
        self.norm = norm
        
        return
    
    def forward(self, input_vector):

        #normalise inputs
        self.input = input_vector

        if self.norm is 'batch':
            self.input = batch_norm(input_vector)
        if self.norm is 'mini_batch':
            self.input = mini_batch_norm(input_vector)
        if self.norm is 'layer':
            self.input = layer_norm(input_vector)

        weighted_sum = self.input @ self.weights 
        output_vector = weighted_sum + self.bias
        
        return output_vector
    
    def back(self, gradient, param):
        
        #initialise optimiser
        if self.optim_available and param['optimiser'] != None:
            self.optim_available = False
            if param['optimiser'] == 'momentum':
                self.optimiser = Momentum(param)
            if param['optimiser'] == 'rms' or 'RMS':
                self.optimiser = RMSprop(param)
            if param['optimiser'] == 'adam':
                self.optimiser = Adam(param)
            else:
                self.optimiser = Gradient_Descent()
                
        
        local_gradient = self.input.unsqueeze(-1)
        gradient = gradient.unsqueeze(-2)
        
        self.delta_w = local_gradient @ gradient
        self.delta_x = (self.weights @ torch.transpose(gradient, -2,-1)).squeeze(-1)

        while len(self.delta_w.size()) > len(self.weights.size()):
            self.delta_w = torch.mean(self.delta_w, dim=0)
            gradient = torch.mean(gradient, dim=0)

        self.weights = self.weights - param['lr'] * self.optimiser.w(self.delta_w)
        self.bias = self.bias - param['lr'] * self.optimiser.b(gradient.squeeze(0))

        return self.delta_x
#__________________________________________________
#CNN LAYERS
#__________________________________________________
class Convolution(object):
    def __init__(self, input_size, kernel_size : tuple = (2,2), stride : int = 1, padding : tuple = (0,0), channels : int = 1, norm=None):
        
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.channels = channels
        self.depth = 1
        self.intial = True
        self.norm = norm
        
        self.size_x = math.floor(((input_size[1] - kernel_size[1] + padding[1] * 2)/stride) + 1)
        self.size_y = math.floor(((input_size[0] - kernel_size[0] + padding[0] * 2)/stride) + 1)
        
        self.kernels = torch.rand((channels, 1,  kernel_size[0], kernel_size[1]), dtype=torch.float) -0.5
        self.bias = torch.zeros(channels, dtype=torch.float)
        
        self.optim_available = True
        self.optimiser = Gradient_Descent()
        
        return
    
    def forward(self, input_matrix): 
        
        #match channels
        if (input_matrix.size()[-3] > 1) and self.intial:
            self.depth = input_matrix.size()[-3]
            self.kernels = torch.rand((self.channels, self.depth, self.kernel_size[0], self.kernel_size[1]), dtype=torch.float) - 0.5
            self.intial = False
             
        #normalise inputs
        if self.norm is 'batch':
            self.input = batch_norm(input_matrix)
        if self.norm is 'mini_batch':
            self.input = mini_batch_norm(input_matrix)
        if self.norm is 'layer':
            self.input = layer_norm(input_matrix)
        
        #pad x
        if self.padding[0] > 0:
            pad_tensor = torch.zeros((input_matrix.size()[-4], input_matrix.size()[-3], input_matrix.size()[-2] + self.padding[0] * 2, input_matrix.size()[-1]), dtype=torch.float)
            pad_tensor[:, :, self.padding[0]: - self.padding[0], :] = input_matrix[:, :, :, :]
            input_matrix = pad_tensor
        
        #inp : 30 x 5 x 30 x 28
        
        #pad y
        if self.padding[1] > 0:
            pad_tensor = torch.zeros((input_matrix.size()[-4], input_matrix.size()[-3], input_matrix.size()[-2], input_matrix.size()[-1] + self.padding[1] * 2), dtype=torch.float)
            pad_tensor[:, :, :, self.padding[1]: - self.padding[1]] = input_matrix[:, :, :, :]
            input_matrix = pad_tensor
            
        #inp : 30 x 5 x 28 x 30
        
        #convolve
        self.input = input_matrix
        self.output_feature = torch.zeros((self.input.size()[-4], self.channels, self.size_y, self.size_x),dtype=torch.float)
        col = 0
        for j in range(self.size_y):
            row = 0
            for i in range(self.size_x):

                window = self.input[:, :, col:col+self.kernel_size[0], row:row+self.kernel_size[1]]
                self.output_feature[:, :, j, i] = torch.sum(window.unsqueeze(1).expand(-1,self.channels,-1,-1,-1) * self.kernels.unsqueeze(0).expand(window.size()[0],-1,-1,-1,-1), dim = (-3,-2,-1)) + self.bias
                row += self.stride                

            col += self.stride

        return self.output_feature
    
    def back(self, gradient, param):
        
        #initialise optimiser
        if self.optim_available and param['optimiser'] != None:
            self.optim_available = False
            if param['optimiser'] == 'momentum':
                self.optimiser = Momentum(param)
            if param['optimiser'] == 'rms' or 'RMS':
                self.optimiser = RMSprop(param)
            if param['optimiser'] == 'adam':
                self.optimiser = Adam(param)
            else:
                self.optimiser = Gradient_Descent()
        
        #inp : 30 x 5 x 28 x 28
        self.delta_k = torch.ones((gradient.size()[-4], self.channels, self.depth, self.kernel_size[0], self.kernel_size[1]),dtype=torch.float)
        grad=gradient.unsqueeze(-3).expand(-1, -1, self.depth, -1,-1)
        
        #calculate kernel gradients
        col = 0
        for j in range(self.kernel_size[0]):
            row = 0
            for i in range(self.kernel_size[1]):

                window = self.input[:, :, col:col+gradient.size()[-2], row:row+gradient.size()[-1]]
                self.delta_k[:, :, :, j, i] = torch.sum(window.unsqueeze(1).expand(-1,self.channels,-1,-1,-1) * grad, dim = (-2,-1))
                row += self.stride
                        
            col += self.stride

        self.delta_x = torch.zeros((gradient.size()[-4], self.depth, self.input_size[0],self.input_size[1]),dtype=torch.float)
        rot_kernal = torch.rot90(self.kernels, 2, [-2, -1])
        
        #calculate input gradients (full convolution)
        padding = (self.input.size()[-2] - self.kernels.size()[-2], self.input.size()[-1] - self.kernels.size()[-1])
        pad_tensor = torch.zeros((rot_kernal.size()[0], rot_kernal.size()[1], rot_kernal.size()[2] + padding[0] * 2, rot_kernal.size()[3] + padding[1] * 2), dtype=torch.float)
        pad_tensor[:, :, padding[0]: -padding[0], padding[1]: - padding[1]] = rot_kernal
        rot_kernal = pad_tensor
        
        for j in range(self.input_size[0]):
            for i in range(self.input_size[1]):
                
                window = rot_kernal[:, :, j:j+gradient.size()[-2], i:i+gradient.size()[-1]]
                self.delta_x[:, :, j, i] += torch.sum(window.unsqueeze(0).expand(grad.size()[0],-1,-1,-1,-1) * grad, dim = (-4, -2,-1))

        #update kernels and biases
        self.kernels -= self.optimiser.w(torch.mean(self.delta_k, dim=0)) * param['lr']
        self.bias -= self.optimiser.b(torch.mean(torch.mean(gradient, dim = (-2,-1)), dim=0)) * param['lr']
        
        #return input grad
        return self.delta_x
#__________________________________________________ 
class Convolution_1D(object):
    def __init__(self, input_size, kernel_size : int = 1, stride : int = 1, padding : int = 0, channels : int = 1, norm = None):
        
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.channels = channels
        self.depth = 1
        self.intial = True
        self.norm = norm
        
        self.size = math.floor(((input_size - kernel_size + padding * 2)/stride) + 1)
        
        self.kernels = torch.rand((channels, 1,  kernel_size), dtype=torch.float) -0.5
        self.bias = torch.zeros(channels, dtype=torch.float) 
        
        self.optim_available = True
        self.optimiser = Gradient_Descent()
        
        return
    
    def forward(self, input_matrix): 
        
        #match channels
        if (input_matrix.size()[-2] > 1) and self.intial:
            self.depth = input_matrix.size()[-2]
            self.kernels = torch.rand((self.channels, self.depth, self.kernel_size), dtype=torch.float) - 0.5
            self.intial = False
             
        #normalise inputs
        if self.norm == 'layer':
            input_matrix = layer_norm(input_matrix)
        if self.norm == 'batch':
            input_matrix = batch_norm(input_matrix)
        
        #pad x
        if self.padding > 0:
            pad_tensor = torch.zeros((input_matrix.size()[-3], input_matrix.size()[-2], input_matrix.size()[-1] + self.padding * 2), dtype=torch.float)
            pad_tensor[:, :, self.padding: - self.padding] = input_matrix[:, :, :]
            input_matrix = pad_tensor
        
        #inp : 30 x 5 x 30 x 28
        
        #convolve
        self.input = input_matrix
        self.output_feature = torch.zeros((self.input.size()[-3], self.channels, self.size),dtype=torch.float)
        col = 0
        #print(self.size)
        for j in range(self.size):
            window = self.input[:, :, col:col+self.kernel_size]
            self.output_feature[:, :, j] = torch.sum(window.unsqueeze(1).expand(-1,self.channels,-1,-1) * self.kernels.unsqueeze(0).expand(window.size()[0],-1,-1,-1), dim = (-2,-1)) + self.bias
            col += self.stride

        return self.output_feature
    
    def back(self, gradient, param):
        
        #initialise optimiser
        if self.optim_available and param['optimiser'] != None:
            self.optim_available = False
            if param['optimiser'] == 'momentum':
                self.optimiser = Momentum(param)
            if param['optimiser'] == 'rms' or 'RMS':
                self.optimiser = RMSprop(param)
            if param['optimiser'] == 'adam':
                self.optimiser = Adam(param)
            else:
                self.optimiser = Gradient_Descent()
        
        #inp : 30 x 5 x 28 x 28
        self.delta_k = torch.ones((gradient.size()[-3], self.channels, self.depth, self.kernel_size),dtype=torch.float)
        grad=gradient.unsqueeze(-2).expand(-1, -1, self.depth, -1)
        
        #calculate kernel gradients
        col = 0
        for j in range(self.kernel_size):
            window = self.input[:, :, col:col+gradient.size()[-1]]
            self.delta_k[:, :, :, j] = torch.sum(window.unsqueeze(1).expand(-1,self.channels,-1,-1) * grad, dim = -1)
            col += self.stride

        self.delta_x = torch.zeros((gradient.size()[-3], self.depth, self.input_size),dtype=torch.float)
        rot_kernal = torch.flip(self.kernels, dims=[-1])
        
        #calculate input gradients (full convolution)
        padding = self.input.size()[-1] - self.kernels.size()[-1]
        pad_tensor = torch.zeros((rot_kernal.size()[0], rot_kernal.size()[1], rot_kernal.size()[2] + padding * 2), dtype=torch.float)
        pad_tensor[:, :, padding: -padding] = rot_kernal
        rot_kernal = pad_tensor
        
        for j in range(self.input_size):        
            window = rot_kernal[:, :, j:j+gradient.size()[-1]]
            self.delta_x[:, :, j] += torch.sum(window.unsqueeze(0).expand(grad.size()[0],-1,-1,-1) * grad, dim = (-3,-1))

        #update kernels and biases
        self.kernels -= self.optimiser.w(torch.mean(self.delta_k, dim=0)) * param['lr']
        self.bias -= self.optimiser.b(torch.mean(torch.mean(gradient, dim = (-2,-1)), dim=0)) * param['lr']
        
        #return input grad
        return self.delta_x
#__________________________________________________   
class Pooling(object):
    def __init__(self, input_size, window_size : tuple = (2,2), stride : int = 1, _type = 'AVG'):
        
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self._type = _type
        
        self.size_x = math.floor(((input_size[1] - window_size[1])/stride) + 1)
        self.size_y = math.floor(((input_size[0] - window_size[0])/stride) + 1)
        
        return
    
    def forward(self, input_matrix): 
        
        self.input = input_matrix
        
        output_feature = torch.zeros((input_matrix.size()[-4], input_matrix.size()[-3], self.size_y, self.size_x),dtype=torch.float)
        self.win_grad = torch.zeros_like(input_matrix)
        y = torch.ones_like(input_matrix)
        
        col = 0
        for j in range(self.size_y):
            row = 0
            for i in range(self.size_x):
                
                window = input_matrix[:, :, col:col+self.window_size[0], row:row+self.window_size[1]]

                if self._type == 'AVG':
                    output_feature[:, :, j, i] = torch.mean(window, dim = (-2,-1))
                    
                else:
                    max_val = torch.amax(window,dim=(-2,-1))
                    output_feature[:, :, j, i] = max_val
                    max_val =torch.where(max_val==0, float('inf'), max_val)
                    win=max_val.unsqueeze(2).unsqueeze(3).repeat_interleave(self.window_size[-1], dim=-1)
                    win=win.repeat_interleave(self.window_size[-2], dim=-2)
                    
                    if torch.count_nonzero(window) != 0:
                        mask = (window == win).float()
                        self.win_grad[:, :, col:col+self.window_size[0], row:row+self.window_size[1]] += mask   

                row += self.stride

            col += self.stride
        
        return output_feature
    
    def back(self, gradient, param):
        
        grad=gradient.repeat_interleave(self.window_size[-1], dim=-1)
        grad=grad.repeat_interleave(self.window_size[-2], dim=-2)

        if self._type == 'AVG':
            window = torch.zeros_like(self.input, dtype=torch.float) + 1/(self.window_size[0] * self.window_size[1])

            col = 0
            for j in range(gradient.size()[-2]):
                row = 0
                for i in range(gradient.size()[-1]):
                    window[:, :, col:col+self.window_size[0], row:row+self.window_size[1]] *= grad[:, :, j * self.window_size[0]:(j * self.window_size[0])+self.window_size[0], i * self.window_size[1]:(i * self.window_size[1])+self.window_size[1]]
                    row += self.stride
                col += self.stride
            grad = window

        else:
            self.win_grad = torch.clamp( self.win_grad, max=1)

            col = 0
            for j in range(gradient.size()[-2]):
                row = 0
                for i in range(gradient.size()[-1]):
                    self.win_grad[:, :, col:col+self.window_size[0], row:row+self.window_size[1]] *= grad[:, :, j * self.window_size[0]:(j * self.window_size[0])+self.window_size[0], i * self.window_size[1]:(i * self.window_size[1])+self.window_size[1]]
                    row += self.stride
                col += self.stride
            grad = self.win_grad
            
        return grad
#__________________________________________________
class Pooling1D(object):
    def __init__(self, input_size, window_size : int = 2, stride : int = 1, _type = 'AVG'):
        
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self._type = _type
        
        return
    
    def forward(self, input_vector):
        return
#__________________________________________________
#TRANFORMER LAYERS
#__________________________________________________
class PostionalEncoding(object):
    def __init__(self):
        return
    
    def encode(self, input_vector):

        batch_size, max_seq_length, d_model = input_vector.size()
        neg = input_vector[:,:,1::2]
        pos = input_vector[:,:,::2]

        index = torch.arange(0, d_model).float()

        neg_index = index[1::2]
        pos_index = index[::2]

        position = torch.arange(0, max_seq_length).float()
        pe_pos = torch.sin(position.unsqueeze(1)/(10000 ** (2 * pos_index/d_model)).unsqueeze(0))
        pe_neg = torch.cos(position.unsqueeze(1)/(10000 ** (2 * neg_index/d_model)).unsqueeze(0))

        pe_pos = pe_pos.unsqueeze(0).expand(batch_size,-1,-1)
        pe_neg = pe_neg.unsqueeze(0).expand(batch_size,-1,-1)

        neg = neg + pe_neg
        pos = pos + pe_pos

        output = torch.zeros_like(input_vector, dtype=torch.float)

        output[:,:,::2] = pos
        output[:,:,1::2] = neg
        
        return output
#__________________________________________________
#ACTIVATION LAYERS
#__________________________________________________
class ReLu(object):
    def __init__(self, maximum_value : float =  float('inf')):
        self.max_val = maximum_value
        return
    
    def forward(self, input_vector):
        
        output = torch.maximum(torch.zeros(1, dtype = torch.float), input_vector)
        output = torch.clamp(output, max=self.max_val)
        self.local_gradient = torch.clamp(output, max=1)
        
        return output
    
    def back(self, gradient, param):
        return self.local_gradient * gradient
#__________________________________________________
class Softmax(object):
    def __init__(self):
        return
    
    def forward(self, input_vector, dim:int=-1):
        self.dim = dim
        output = torch.div(torch.exp(input_vector), torch.sum(torch.exp(input_vector), dim=dim).unsqueeze(dim))
        self.s = output

        return output
    
    def back(self, gradient, param):
        
        identity = torch.diag_embed(self.s)
        ss = self.s.unsqueeze(self.dim) @ self.s.unsqueeze(self.dim-1)
        grad = identity - ss

        gradient = gradient.unsqueeze(-1)
        out = (grad @ gradient).squeeze(-1)
            
        return out
#__________________________________________________
class Sigmoid(object):
    def __init__(self):
        return
    
    def forward(self, input_vector):
        output = 1/(1 + torch.exp(-input_vector))
        self.sig = output
        return output
    
    def back(self, gradient, param):
        
        grad = (self.sig * (1 - self.sig)) * gradient

        return grad 
#__________________________________________________
#OPTIMISERS
#__________________________________________________
class Momentum(object):
    def __init__(self, param):
        self.average_b = 0
        self.average_w = 0
        self.momentum = param['optimiser_lr']
        return
    
    def w(self, gradient):
        self.average_w = self.momentum * self.average_w + (1-self.momentum) * gradient
        return self.average 
    
    def b(self, gradient):
        self.average_b = self.momentum * self.average_b + (1-self.momentum) * gradient
        return self.average 
#__________________________________________________
class RMSprop(object):
    def __init__(self, param):
        self.s_dW = 0
        self.s_dB = 0
        self.beta = param['optimiser_lr']
        return
    
    def w(self, gradient):
        self.s_dW = self.beta * self.s_dW + (1 - self.beta) * (gradient ** 2)
        return gradient / (self.s_dW ** (1/2) + 1e-8)
    
    def b(self, gradient):
        self.s_dB = self.beta * self.s_dB + (1 - self.beta) * (gradient ** 2)
        return gradient / (self.s_dB ** (1/2) + 1e-8)
#__________________________________________________
class Adam(object):
    def __init__(self, param):
        self.s_dW = 0
        self.s_dB = 0
        self.average_b = 0
        self.average_w = 0
        self.beta_1 = param['optimiser_b1']
        self.beta_2 = param['optimiser_b2']  
        return
    
    def w(self, gradient):
        self.average_w = self.beta_1 * self.average_w + (1-self.beta_1) * gradient
        self.s_dW = self.beta_2 * self.s_dW + (1 - self.beta_2) * (gradient ** 2)
        return self.average_w / (self.s_dW ** (1/2) + 1e-8)
    
    def b(self, gradient):
        self.average_b = self.beta_1 * self.average_b + (1-self.beta_1) * gradient
        self.s_dB = self.beta_2 * self.s_dB + (1 - self.beta_2) * (gradient ** 2)
        return self.average_b / (self.s_dB ** (1/2) + 1e-8)
#__________________________________________________
class Gradient_Descent(object):
    def __init__(self):
        return
    
    def w(self, gradient):
        return gradient
    
    def b(self, gradient):
        return gradient
#__________________________________________________
#RANDOM
#__________________________________________________
class Batch_Flatten(object):
    def __init__(self):
        self.i_shape = 0
        return
    
    def forward(self, input_vector):
        self.i_shape = input_vector.size()
        output = input_vector.view(input_vector.size()[0], -1)
        return output
    
    def back(self, input_vector, param):
        return torch.reshape(input_vector, self.i_shape)