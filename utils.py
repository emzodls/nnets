from random import random
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def split_set(data,fractionTraining):

    train = set()
    test = set()
    dice = random()

    if dice <= fractionTraining:
        train.add(data.pop())
    else:
        test.add(data.pop())

    return train,test



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

