import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import os
from rnn import RNN

from random import shuffle

from utils import split_set,lineToTensor

categories = {0:'not word',1:'word'}

def prepareTensors(labeled_pair):
    sequence,label = labeled_pair
    category_tensor = torch.tensor([label], dtype=torch.long)
    line_tensor = lineToTensor(sequence)
    category = categories[label]
    return category, sequence, category_tensor, line_tensor


def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0].item()
    return categories[category_i], category_i

PATH_TO_SETS = '/Volumes/lab_data/language_training'

languages = ['en','fr','de','it','es','pt']

negatives = '/Volumes/lab_data/language_training/negatives.txt'


## Select Language to Exclude Randomly
## first generate the training set

language_excluded = 'en'

# Only take 60% of the negative test set for training

negativeSet = set(open('/Volumes/lab_data/language_training/negatives.txt').read().strip().split(','))

negativeTrainSet,negativeTestSet = split_set(negativeSet,0.6)

positives = set()

for language in languages:
    if language != language_excluded:
        positives.update(open(os.path.join(PATH_TO_SETS,language+'.txt')).read().strip().split(','))


trainingSet = [(neg, 0) for neg in negativeTrainSet]
trainingSet.extend((pos,1) for pos in positives)

print('Training with {} positives and {} negatives, for a total of {} exemplars'.format(len(positives),
                            len(negativeTrainSet),len(trainingSet)))

n_epochs = 30
n_iters = 100000
print_every = 5000
plot_every = 1000
frac_train = 0.95

n_hidden = 256

learning_rate = 0.01

model = RNN(26,n_hidden,2)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)



def train(category_tensor,line_tensor):
    model.zero_grad()
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden  = model(line_tensor[i],hidden)

    loss = criterion(output,category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


def evaluate(line_tensor):
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

trainingIdx = math.floor(len(trainingSet) * frac_train)

all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(n_epochs):
    ## Shuffle Training Set Every Epoch and Save 5% for validation
    print('Starting Epoch {}'.format(epoch+1))
    shuffle(trainingSet)
    numberTrained = len(trainingSet[:trainingIdx])
    current_loss = 0
    for idx,labeled_pair in enumerate(trainingSet[:trainingIdx]):
        category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
        output, loss = train(category_tensor,line_tensor)
        current_loss += loss

        if idx % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                idx, idx / numberTrained * 100, timeSince(start), loss, line, guess, correct))

    print('Epoch {} finished, Average Loss = {}'.format(epoch + 1,current_loss/numberTrained))
    all_losses.append(current_loss/numberTrained)
    print('Testing Model with Validation Data.')

    with torch.no_grad():
        correct_word = 0
        total_word = 0
        correct_not_word = 0
        total_not_word = 0
        for labeled_pair in trainingSet[trainingIdx:]:
            category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
            output = evaluate(line_tensor)
            guess = category_from_output(output)
            if category == 1:
                total_word += 1
                if guess == 1:
                    correct_word += 1
            else:
                total_not_word += 1
                if guess == 0:
                    correct_not_word += 1
        print('Epoch {}, % Correct Word = {}, % Correct Not Word = {}, Total Score = {}'.format(epoch+1,
                                                                                                (correct_word/total_word)*100,
                                                                                                (correct_not_word / total_not_word) * 100,
                                                                                                ((correct_word+ correct_not_word) / (total_not_word+total_word)) * 100))


torch.save(model.state_dict(), 'model.ckpt')