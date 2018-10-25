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

PATH_TO_SETS = '/home/ubuntu/training_data'

languages = ['en','fr','de','it','es','pt']

negatives = '/home/ubuntu/training_data/negatives.txt'


## Select Language to Exclude Randomly
## first generate the training set

language_excluded = 'en'

# Only take 60% of the negative test set for training

negativeSet = set(open('/home/ubuntu/training_data/negatives.txt').read().strip().split(','))

negativeTrainSet,negativeTestSet = split_set(negativeSet,0.6)

positives = set()

for language in languages:
    if language != language_excluded:
        positives.update(open(os.path.join(PATH_TO_SETS,language+'.txt')).read().strip().split(','))


masterSet = [(neg, 0) for neg in negativeTrainSet]
masterSet.extend((pos,1) for pos in positives)


#shuffle(trainingSet)

## Test with only 30% of training set initially for testing

#trainingSet = trainingSet[:math.floor(len(trainingSet)*0.1)]

n_epochs = 10
print_every = 100000
plot_every = 1000
frac_train = 0.95

n_hidden = 256

learning_rate = 0.0005

model = RNN(26,n_hidden,2)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8,nesterov=True)

if os.path.isfile('model_SGD_epoch_9.ckpt'):
    checkpoint = torch.load('model_SGD_epoch_9.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("=> loaded checkpoint ")
    with open('logfile.log','a') as outfile:
        outfile.write("=> loaded checkpoint\n")

#optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer = torch.optim.ASGD(model.parameters(),lr=learning_rate)

def train(category_tensor,line_tensor):
    model.zero_grad()
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden  = model(line_tensor[i],hidden)

    loss = criterion(output,category_tensor)
    loss.backward()
    params = list(model.parameters()) + list(criterion.parameters())
    torch.nn.utils.clip_grad_norm_(params,0.2)
    optimizer.step()

    return output, loss.item()

def validate(category_tensor,line_tensor):
    model.eval()
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden  = model(line_tensor[i],hidden)
    loss = criterion(output,category_tensor)
    return output, loss.item()

def evaluate(line_tensor):
    model.eval()
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output

trainingIdx = math.floor(len(masterSet) * frac_train)

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
    shuffle(masterSet)
    trainingSet = masterSet
    print('Training with {} positives and {} negatives, for a total of {} exemplars'.format(len(positives),
                                                                                            len(negativeTrainSet),
                                                                                            len(trainingSet)))
    with open('logfile.log', 'a') as outfile:
        outfile.write('Starting Epoch {}\n'.format(epoch+1))
        outfile.write('Training with {} positives and {} negatives, for a total of {} exemplars\n'.format(len(positives),
                                                                                            len(negativeTrainSet),
                                                                                            len(trainingSet)))
    trainingIdx = math.floor(len(trainingSet) * frac_train)
    numberTrained = len(trainingSet[:trainingIdx])
    current_loss = 0
    for idx,labeled_pair in enumerate(trainingSet[:trainingIdx]):
        model.train()
        category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
        output, loss = train(category_tensor,line_tensor)
        current_loss += loss

        if idx % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                idx, idx / numberTrained * 100, timeSince(start), loss, line, guess, correct))
            with open('logfile.log', 'a') as outfile:
                outfile.write('%d %d%% (%s) %.4f %s / %s %s\n' % (
                    idx, idx / numberTrained * 100, timeSince(start), loss, line, guess, correct))

    print('Epoch {} finished, Average Loss = {}'.format(epoch + 1,current_loss/numberTrained))
    all_losses.append(current_loss/numberTrained)
    print('Testing Model with Validation Data.')

    with open('logfile.log', 'a') as outfile:
        outfile.write('Epoch {} finished, Average Loss = {}\n'.format(epoch + 1,current_loss/numberTrained))
        outfile.write('Testing Model with Validation Data.\n')
    with torch.no_grad():
        correct_word = 0
        total_word = 0
        correct_not_word = 0
        total_not_word = 0
        total_val_loss = 0
        numValSet = len(trainingSet[trainingIdx:])
        print('Validating with {} samples'.format(len(trainingSet[trainingIdx:])))
        for labeled_pair in trainingSet[trainingIdx:]:
            category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
            output,val_loss = validate(category_tensor,line_tensor)
            guess = category_from_output(output)
            total_val_loss += loss
            #print(category,guess)
            if category == 'word':
                total_word += 1
                if guess[0] == category:
                    correct_word += 1
            else:
                total_not_word += 1
                if guess[0] == category:
                    correct_not_word += 1
        print(correct_not_word,correct_word)
        print('Epoch {}, Correct Word = {} %, Correct Not Word = {} %, Total Score = {}'.format(epoch+1,
                                                                                                (correct_word/total_word)*100,
                                                                                                (correct_not_word / total_not_word) * 100,
                                                                                                ((correct_word+ correct_not_word) / (total_not_word+total_word)) * 100))
        print('Average Loss = {}'.format(total_val_loss/numValSet))

        with open('logfile.log', 'a') as outfile:
            outfile.write('Validating with {} samples\n'.format(len(trainingSet[trainingIdx:])))
            outfile.write('Epoch {}, Correct Word = {} %, Correct Not Word = {} %, Total Score = {}\n'.format(epoch + 1,
                                                                                                    (
                                                                                                                correct_word / total_word) * 100,
                                                                                                    (
                                                                                                                correct_not_word / total_not_word) * 100,
                                                                                                    ((
                                                                                                                 correct_word + correct_not_word) / (
                                                                                                                 total_not_word + total_word)) * 100))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},'model_SGD_epoch_{}.ckpt'.format(epoch))