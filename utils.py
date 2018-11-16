from random import random
import torch

letters = 'ACDEFGHIKLMNPQRSTVWY'
letterToIdx = dict(zip(letters,range(20)))
categories = {0:'not word',1:'word'}

def split_set(data,fractionTraining):
    train = set()
    test = set()
    while data:
        dice = random()
        element = data.pop()
        if element:
            if dice <= fractionTraining:
                train.add(element)
            else:
                test.add(element)

    return train,test


def letterToIndex(letter):
    return letterToIdx[letter]

def lineToTensor(line):
    #line = line.lower()
    tensor = torch.zeros(len(line), 1, 20)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor



def prepareTensors(labeled_pair):
    sequence,label = labeled_pair
    category_tensor = torch.tensor([label], dtype=torch.long)
    line_tensor = lineToTensor(sequence)
    category = categories[label]
    return category, sequence, category_tensor, line_tensor

def evaluate(model,line_tensor):
    model.eval()
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output
