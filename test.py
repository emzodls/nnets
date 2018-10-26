import torch
import torch.nn as nn
import time
import math
import os
from rnn import RNN

from random import shuffle

from utils import split_set,lineToTensor,categories,evaluate

categories = {0:'not word',1:'word'}

def validate(category_tensor,line_tensor):
    model.eval()
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden  = model(line_tensor[i],hidden)
    loss = criterion(output,category_tensor)
    return output, loss.item()

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

PATH_TO_SETS = '/Users/emzodls/Documents/training_data'

languages = ['en','fr','de','it','es','pt']

negatives = '/Users/emzodls/Documents/negatives.txt'


## Select Language to Exclude Randomly
## first generate the training set

language_excluded = 'en'

# Only take 60% of the negative test set for training

negativeSet = set(open('/Users/emzodls/Documents/training_data/negatives.txt').read().strip().split(','))

negativeTrainSet,negativeTestSet = split_set(negativeSet,0.6)

positives = set()
test_langauge = set()

for language in languages:
    if language != language_excluded:
        positives.update(open(os.path.join(PATH_TO_SETS,language+'.txt')).read().strip().split(','))
    else:
        test_langauge.update(open(os.path.join('/Users/emzodls/Documents/en_dict.txt')).read().strip().split(','))


test_langauge = test_langauge - positives
masterSet = [(neg, 0) for neg in negativeTrainSet]
masterSet.extend((pos,1) for pos in positives)

testSet = [(neg,0) for neg in negativeTestSet]
testSet.extend((pos,1) for pos in test_langauge)

shuffle(masterSet)
shuffle(testSet)

n_hidden = 256

saved_model_path = '/Users/emzodls/model_entire_set.ckpt'

model = RNN(26,n_hidden,2)
learning_rate = 0.0005

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8,nesterov=True)

if os.path.isfile(saved_model_path):
    checkpoint = torch.load('model_SGD_epoch_9.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("=> loaded checkpoint ")
    with open('logfile.log','a') as outfile:
        outfile.write("=> loaded checkpoint\n")

print('Testing on Entire Set')
print_every = 10000

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

with torch.no_grad():
    wrong_words = set()
    print('Testing on {} English Words'.format(len(test_langauge)))
    correct_word = 0
    total_word = 0
    correct_not_word = 0
    total_not_word = 0
    total_val_loss = 0
    for idx,labeled_pair in enumerate((pos,1) for pos in test_langauge):
        category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
        output = evaluate(model,line_tensor)
        guess = category_from_output(output)
        if category == 'word':
            total_word += 1
            if guess[0] == category:
                correct_word += 1
            else:
                wrong_words.add(line)
        else:
            total_not_word += 1
            if guess[0] == category:
                correct_not_word += 1
        if idx % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %s / %s %s' % (
                idx, idx / len(test_langauge) * 100, timeSince(start), line, guess, correct))

    print('Correct Word = {} %'.format((correct_word / total_word) * 100))

    with open('wrong_words.txt','w') as outfile:
        outfile.write(','.join(wrong_words))
    correct_word = 0
    total_word = 0
    correct_not_word = 0
    total_not_word = 0
    total_val_loss = 0
    numValSet = len(masterSet)
    print('Testing with {} samples'.format(len(masterSet)))
    for idx,labeled_pair in enumerate(masterSet):
        category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
        output = evaluate(model,line_tensor)
        guess = category_from_output(output)
        # print(category,guess)
        if category == 'word':
            total_word += 1
            if guess[0] == category:
                correct_word += 1
        else:
            total_not_word += 1
            if guess[0] == category:
                correct_not_word += 1
        if idx % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %s / %s %s' % (
                idx, idx / numValSet * 100, timeSince(start), line, guess, correct))
    print('Correct Word = {} %, Correct Not Word = {} %, Total Score = {}'.format((
                                                                                                        correct_word / total_word) * 100,
                                                                                            (
                                                                                                        correct_not_word / total_not_word) * 100,
                                                                                            ((
                                                                                                         correct_word + correct_not_word) / (
                                                                                                         total_not_word + total_word)) * 100))
    print('Average Loss = {}'.format(total_val_loss / numValSet))

    print('Testing on English Words')
    print('With {} positives and {} negatives, for a total of {} exemplars'.format(len(testSet),
                                                                                            len(negativeTestSet),
                                                                                            len(testSet)))
    correct_word = 0
    total_word = 0
    correct_not_word = 0
    total_not_word = 0
    total_val_loss = 0
    for idx,labeled_pair in enumerate(testSet):
        category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
        output = evaluate(model,line_tensor)
        guess = category_from_output(output)
        if category == 'word':
            total_word += 1
            if guess[0] == category:
                correct_word += 1
        else:
            total_not_word += 1
            if guess[0] == category:
                correct_not_word += 1
        if idx % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %s / %s %s' % (
                idx, idx / len(testSet) * 100, timeSince(start), line, guess, correct))

    print('Correct Word = {} %, Correct Not Word = {} %, Total Score = {}'.format((
                                                                                                        correct_word / total_word) * 100,
                                                                                            (
                                                                                                        correct_not_word / total_not_word) * 100,
                                                                                            ((
                                                                                                         correct_word + correct_not_word) / (
                                                                                                         total_not_word + total_word)) * 100))

