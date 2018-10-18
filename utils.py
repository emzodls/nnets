from random import random
import torch

letters = 'abcdefghijklmnopqrstuvwxyz'
letterToIdx = dict(zip(letters,range(26)))

def split_set(data,fractionTraining):

    train = set()
    test = set()
    dice = random()

    if dice <= fractionTraining:
        train.add(data.pop())
    else:
        test.add(data.pop())

    return train,test


def letterToIndex(letter):
    return letterToIdx[letter]

def lineToTensor(line):
    line = line.lower()
    tensor = torch.zeros(len(line), 1, 26)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

