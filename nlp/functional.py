import sys
sys.path.append('..')
import re
from scipy.stats import skewnorm
import math
import numpy as np
import torch

EMBEDDINGS = "C:\\Users\\jugwu\\Documents\\justins_work\\CNN\\Transformers\\Embedding\\"
DATAPATH = "C:\\Users\\jugwu\\Documents\\justins_work\\CNN\\cleanData\\"
PADDING = '<pad>'
START = '<start>'
END = '<end>'

#__________________________________________________
def load_data(file, num:int=1, max_sequence_length:int=10):
    with open(DATAPATH + file + ".txt", 'r',encoding="utf-8") as f:
        lines = f.readlines()

    #print(lines)

    data_points = []
    data_labels = []

    for i, sentence in enumerate(lines):
        if i >= num:
            break

        sentence = re.sub("\n", "", sentence)

        words = sentence.split(' ')
        words.append(END)

        pad = [PADDING] * (len(words) + (max_sequence_length - 1) * 2)
        pad[max_sequence_length:-max_sequence_length] = words
        pad[max_sequence_length-1] = START

        for j, word in enumerate(words):
            if j > 0:
                data_labels.append(word)
                data_points.append(pad[j:j + max_sequence_length])

    #for data, label in zip(data_points, data_labels):
    #    print(f"data: {data}, label: {label}")
        
    return data_points, data_labels
#__________________________________________________
def tokenise(file, num:int=2):
    with open(DATAPATH + file + ".txt", 'r',encoding="utf-8") as f:
        lines = f.readlines()

    text = ' '.join(lines[:num])
    words = text.split(' ')

    vocab = []
    tokens = []

    tag = 0
    for word in words:
        word = re.sub('\n', '', word)
        if word not in vocab:
            vocab.append(word)
            tokens.append(tag)
            tag+=1

    return tokens, vocab
#__________________________________________________
def skew(num):
    num_locs = num

    numValues = 10000
    maxValue = ((num_locs * 2) - 1)
    skewness = 0  #Negative values are left skewed, positive values are right skewed.

    random = skewnorm.rvs(a = skewness,loc=maxValue, size=numValues)  #Skewnorm function

    random = random - min(random)      #Shift the set so the minimum value is equal to zero.
    random = random / max(random)      #Standadize all the vlues between 0 and 1. 
    random = random * maxValue   
    random = random - (num_locs)      #Multiply the standardized values by the maximum value.
    
    list_ = []
    for i in range(numValues):
        index = np.random.randint(numValues)
        num = random[index]
        number = math.floor(num)
        if number == 0:
            number = np.random.randint(maxValue + 1) - num_locs 
        if number == 0:
            if np.random.rand() < 0.5:
                number = 1 
            else:
                number = -1
        list_.append(math.floor(number))
        
    
    max_index = np.shape(list_)[0]
    sum_ = np.zeros(num_locs * 2)
    
    for num_ in list_:
        sum_[num_+num_locs] += 1

    return list_, max_index
#__________________________________________________
#__________________________________________________
def load_vectors(file):
    with open(EMBEDDINGS + file + "_vocab.txt", 'r',encoding="utf-8") as f:
        lines = f.readlines()

    temp = []
    vector = []
    count = 0
    for i, line in enumerate(lines):
        line = re.sub("\n", "", line)
        if i == 0:
            size = int(line)
        else:
            temp.append(float(line))
            if count == size - 1:
                vector.append(torch.tensor(temp, dtype=torch.float))
                count=-1
                temp = []
            count+=1

    vector = torch.stack(vector)
    #print(vector.size())

    vocab = []
    with open(EMBEDDINGS + file + "_tokens.txt", 'r',encoding="utf-8") as f:
        lines = f.readlines()
        
        for line in lines:
            line = re.sub("\n", "", line)
            vocab.append(line)

    #print(np.shape(vocab))
            
    return vector, vocab
#__________________________________________________
#__________________________________________________
#__________________________________________________
#__________________________________________________
#__________________________________________________