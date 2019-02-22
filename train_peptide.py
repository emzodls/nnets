import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import os
from rnn import RNN

from random import shuffle

from utils import split_set,lineToTensor,prepareTensors,categories


def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0].item()
    return categories[category_i], category_i

PATH_TO_SETS = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets'
amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

negatives = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/negatives-trim.txt'
negative_lanthi = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/lanthi-neg.txt'

positive_files = ['lanthi-all-clean','lasso-as4-clean','thio-all-clean','ripper-lasso-rodeo-all','ripper-micro-rodeo-all',
                  'ripper-thio-rodeo-all','sacti-as4-clean','mcgarvey','ripper-hmm-pos']

logfile_name = 'logfile_all_lanthi_neg.log'

positives = set()

for pos_training in positive_files:
    for line in open(os.path.join(PATH_TO_SETS,pos_training+'.txt')):
        line=line.strip().upper()
        if all(x in amino_acids for x in line):
            positives.add(line)

# Only take 50% of the negative test set for training

negativeSet = set()

for line in open(negatives):
    line=line.strip().upper()
    if all(x in amino_acids for x in line):
        negativeSet.add(line)

negativeLantiSet = set()
for line in open(negative_lanthi):
    line=line.strip().upper()
    if line not in positives:
        if all(x in amino_acids for x in line):
            negativeLantiSet.add(line)

positives = set(x for x in positives if len(x) >= 20 and len(x) <= 120)
negativeSet = set(x for x in negativeSet if len(x) >= 20 and len(x) <= 120)
negativeLantiSet = set(x for x in negativeLantiSet if len(x) >= 20 and len(x) <= 120)

negativeTrainSet,negativeTestSet = split_set(negativeSet,0.50)
negativeLantiTrainSet,negativeLantiTestSet = split_set(negativeLantiSet,0.50)

print('Total Number of positives: {}'.format(len(positives)))
print('Total Number of negatives: {}'.format(len(negativeTrainSet|negativeLantiTrainSet)))

masterSet = [(neg, 0) for neg in negativeTrainSet]
masterSet.extend((neg, 0) for neg in negativeLantiTrainSet)
masterSet.extend((pos,1) for pos in positives)


#shuffle(trainingSet)

## Test with only 30% of training set initially for testing

#trainingSet = trainingSet[:math.floor(len(trainingSet)*0.1)]

n_epochs = 200
print_every = 500
frac_train = 0.90

n_hidden = 512

learning_rate = 0.001

model = RNN(20,n_hidden,2)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8,nesterov=True)

if os.path.isfile('all_pos_148.ckpt'):
    checkpoint = torch.load('all_pos_148.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("=> loaded checkpoint ")
    with open(logfile_name,'a') as outfile:
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
    negativeTrainSet, negativeTestSet = split_set(negativeTrainSet|negativeTestSet, 0.50)
    negativeLantiTrainSet, negativeLantiTestSet = split_set(negativeLantiTrainSet|negativeLantiTestSet, 0.50)
    print('Total Number of positives: {}'.format(len(positives)))
    print('Total Number of negatives: {}'.format(len(negativeTrainSet | negativeLantiTrainSet)))
    masterSet = [(neg, 0) for neg in negativeTrainSet]
    masterSet.extend((neg, 0) for neg in negativeLantiTrainSet)
    masterSet.extend((pos, 1) for pos in positives)
    shuffle(masterSet)
    trainingSet = masterSet
    print('Training with {} positives and {} negatives, for a total of {} exemplars'.format(len(positives),
                                                                                            len(negativeTrainSet|negativeLantiTrainSet),
                                                                                            len(trainingSet)))
    with open(logfile_name, 'a') as outfile:
        outfile.write('Starting Epoch {}\n'.format(epoch+1 ))
        outfile.write('Training with {} positives and {} negatives, for a total of {} exemplars\n'.format(len(positives),
                                                                                            len(negativeTrainSet|negativeLantiTrainSet),
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
                idx, idx / numberTrained * 100, timeSince(start), current_loss/(idx+1), line, guess, correct))
            with open(logfile_name, 'a') as outfile:
                outfile.write('%d %d%% (%s) %.4f %s / %s %s\n' % (
                    idx, idx / numberTrained * 100, timeSince(start), current_loss/(idx+1), line, guess, correct))

    print('Epoch {} finished, Average Loss = {}'.format(epoch + 1,current_loss/numberTrained))
    all_losses.append(current_loss/numberTrained)
    print('Testing Model with Validation Data.')

    with open(logfile_name, 'a') as outfile:
        outfile.write('Epoch {} finished, Average Loss = {}\n'.format(epoch + 1 ,current_loss/numberTrained))
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
        print('Epoch {}, Correct Word = {} %, Correct Not Word = {} %, Total Score = {}, Total Word = {}, Total Not Word = {}'.format(epoch+1,
                                                                                                (correct_word/total_word)*100,
                                                                                                (correct_not_word / total_not_word) * 100,
                                                                                                ((correct_word+ correct_not_word) / (total_not_word+total_word)) * 100,total_word,total_not_word))
        print('Average Loss = {}'.format(total_val_loss/numValSet))

        with open(logfile_name, 'a') as outfile:
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
            'loss': loss},'all_pos_lanthi_neg_{}.ckpt'.format(epoch))