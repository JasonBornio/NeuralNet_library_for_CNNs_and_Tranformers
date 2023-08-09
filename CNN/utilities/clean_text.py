# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:54:47 2023

@author: jugwu
"""
import os.path
import re
import string
from tqdm import tqdm
import numpy as np
RAWPATH = "C:\\Users\\jugwu\\Documents\\justins_work\\CNN\\data\\textDATA\\"
DATAPATH = "C:\\Users\\jugwu\\Documents\\justins_work\\CNN\\cleanData\\"
def clean(file, file2, num:int=100000):
    with open(RAWPATH + file + ".txt", 'r',encoding="utf-8") as f:
        lines = f.readlines()
    
    data =[]
    print("processing....")
    for line in tqdm(lines):
        long = line
        #clean text
        long = re.sub('\n','',long)
        long = re.sub('  ',' ',long)
        long = re.sub('  ',' ',long)
        long = re.sub(r"{.*}", "{{}}", long)
        long = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", long)
        long = re.sub("\(", "",long)
        long = re.sub("\)", "",long)
        
        
        if long != '\n' and long != '':
            long = long.split('. ')
            for word in long:
                word = word.split('" ')
                for w in word:
                    data.append(w)
                    
        
    
    clean = []
    for sentence in data:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
 
        new_sentence = []
        for word in sentence.split(' '):
            word = re.sub('Ill', "i'll", word)
            word = re.sub('cant', "can't", word.lower())
            word = re.sub('dont', "don't", word.lower())
            word = re.sub('wont', "won't", word.lower())
            word = re.sub('its', "it's", word.lower())
            word = re.sub('doesnt', "dosen't", word.lower())
            word = re.sub('arent', "aren't", word.lower())
            word = re.sub('shouldnt', "shouldn't", word.lower())
            word = re.sub('couldnt', "couldn't", word.lower())
            word = re.sub('didnt', "didn't", word.lower())
            word = re.sub('wouldnt', "wouldn't", word.lower())
            new_sentence.append(word)
            
        clean.append(' '.join(new_sentence))
        
    clean2 = [] 
    print("processing....")
    for line in tqdm(clean):
        long = line
        #clean text
        long = re.sub('\n','',long)
        long = re.sub('  ',' ',long)
        long = re.sub('  ',' ',long)  
        if long != '\n' and long != '' and np.shape(long.split(' '))[0] > 3:
            clean2.append(long)
        
    print("writing......")
    with open(DATAPATH + file2 + ".txt", 'w',encoding="utf-8") as f:
        for line in tqdm(clean2):
            f.write(line)
            f.write("\n")

    print()
    return 