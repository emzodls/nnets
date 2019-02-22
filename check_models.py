import torch
import torch.nn as nn
import time
import math
import os
from rnn import RNN
from glob import glob
from Bio import SeqIO

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


PATH_TO_SETS = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets'
amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
test_set_folder = '/Volumes/lab_data/Ripps/training_sets/test_sets'

negatives = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/negatives-trim.txt'
negative_lanthi = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/lanthi-neg.txt'

positive_files = ['lanthi-all-clean','lasso-as4-clean','thio-all-clean','ripper-lasso-rodeo-all','ripper-micro-rodeo-all',
                  'ripper-thio-rodeo-all','sacti-as4-clean','mcgarvey','ripper-hmm-pos']

logfile_name = '/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/check_models.log'

positives = set()

for pos_training in positive_files:
    for line in open(os.path.join(PATH_TO_SETS,pos_training+'.txt')):
        line=line.strip().upper()
        if all(x in amino_acids for x in line):
            positives.add(line)

# Only take 60% of the negative test set for training

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

# negativeTrainSet,negativeTestSet = split_set(negativeSet,0.50)
# negativeLantiTrainSet,negativeLantiTestSet = split_set(negativeLantiSet,0.50)

negatives = negativeSet|negativeLantiSet

print('Testing on Entire Set')
print_every = 1000

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

with open(logfile_name, 'a') as outfile:
    outfile.write('Num Positive: {}, Num Negative: {}\n'.format(len(positives), len(negatives)))
print('Num Positive: {}, Num Negative: {}'.format(len(positives), len(negatives)))
with torch.no_grad():
    #models_to_test = ['all_pos_{}.ckpt'.format(x) for x in range(160)]
    models_to_test = ['all_pos_lanthi_neg_{}.ckpt'.format(x) for x in range(99,100)]
    for model_name in models_to_test:
        n_epochs = 30
        print_every = 100
        frac_train = 0.95

        n_hidden = 512

        learning_rate = 0.001

        model = RNN(20, n_hidden, 2)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, nesterov=True)
        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded {}".format(model_name))
            with open(logfile_name, 'a') as outfile:
                outfile.write("=> loaded {}\n".format(model_name))
        correct_word = 0
        total_word = 0
        correct_not_word = 0
        total_not_word = 0
        total_val_loss = 0
        for idx,labeled_pair in enumerate((pos,1) for pos in positives):
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
        for idx,labeled_pair in enumerate((neg,0) for neg in negatives):
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
        print('Model: {}, Correct Word = {:.4f} %, Correct Not Word = {:.4f} % Total Score = {:.4f} %'.format(model_name,
                                                                               (correct_word / total_word) * 100,
                                                                            (correct_not_word / total_not_word) * 100,
              (correct_word + correct_not_word) / (total_not_word + total_word) * 100))

        with open(logfile_name, 'a') as outfile:
            outfile.write('Model: {}, Correct Word = {:.4f} %, Correct Not Word = {:.4f} % Total Score = {:.4f} %\n'.format(model_name,
                                                                               (correct_word / total_word) * 100,
                                                                            (correct_not_word / total_not_word) * 100,
              (correct_word + correct_not_word) / (total_not_word + total_word) * 100))
    #
    # with open('wrong_words.txt','w') as outfile:
    #     outfile.write(','.join(wrong_words))
    # correct_word = 0
    # total_word = 0
    # correct_not_word = 0
    # total_not_word = 0
    # total_val_loss = 0
    # numValSet = len(masterSet)
    # print('Testing with {} samples'.format(len(masterSet)))
    # for idx,labeled_pair in enumerate(masterSet):
    #     category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
    #     output = evaluate(model,line_tensor)
    #     guess = category_from_output(output)
    #     # print(category,guess)
    #     if category == 'word':
    #         total_word += 1
    #         if guess[0] == category:
    #             correct_word += 1
    #     else:
    #         total_not_word += 1
    #         if guess[0] == category:
    #             correct_not_word += 1
    #     if idx % print_every == 0:
    #         guess, guess_i = category_from_output(output)
    #         correct = '✓' if guess == category else '✗ (%s)' % category
    #         print('%d %d%% (%s) %s / %s %s' % (
    #             idx, idx / numValSet * 100, timeSince(start), line, guess, correct))
    # print('Correct Word = {} %, Correct Not Word = {} %, Total Score = {}'.format((
    #                                                                                                     correct_word / total_word) * 100,
    #                                                                                         (
    #                                                                                                     correct_not_word / total_not_word) * 100,
    #                                                                                         ((
    #                                                                                                      correct_word + correct_not_word) / (
    #                                                                                                      total_not_word + total_word)) * 100))
    # print('Average Loss = {}'.format(total_val_loss / numValSet))
    #
    # print('Testing on English Words')
    # print('With {} positives and {} negatives, for a total of {} exemplars'.format(len(testSet),
    #                                                                                         len(negativeTestSet),
    #                                                                                         len(testSet)))
    # correct_word = 0
    # total_word = 0
    # correct_not_word = 0
    # total_not_word = 0
    # total_val_loss = 0
    # for idx,labeled_pair in enumerate(testSet):
    #     category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
    #     output = evaluate(model,line_tensor)
    #     guess = category_from_output(output)
    #     if category == 'word':
    #         total_word += 1
    #         if guess[0] == category:
    #             correct_word += 1
    #     else:
    #         total_not_word += 1
    #         if guess[0] == category:
    #             correct_not_word += 1
    #     if idx % print_every == 0:
    #         guess, guess_i = category_from_output(output)
    #         correct = '✓' if guess == category else '✗ (%s)' % category
    #         print('%d %d%% (%s) %s / %s %s' % (
    #             idx, idx / len(testSet) * 100, timeSince(start), line, guess, correct))
    #
    # print('Correct Word = {} %, Correct Not Word = {} %, Total Score = {}'.format((
    #                                                                                                     correct_word / total_word) * 100,
    #                                                                                         (
    #                                                                                                     correct_not_word / total_not_word) * 100,
    #                                                                                         ((
    #                                                                                                      correct_word + correct_not_word) / (
    #                                                                                                      total_not_word + total_word)) * 100))
    #
