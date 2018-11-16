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

negatives = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/negatives-trim.txt'
negative_lanthi = '/Volumes/lab_data/Ripps/training_sets/cleaned_sets/lanthi-neg.txt'

positive_files = ['lanthi-all-clean','lasso-as4-clean','thio-all-clean']

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
for line in open(negatives):
    line=line.strip().upper()
    if line not in positives:
        if all(x in amino_acids for x in line):
            negativeLantiSet.add(line)

negativeTrainSet,negativeTestSet = split_set(negativeSet,0.35)
negativeLantiTrainSet,negativeLantiTestSet = split_set(negativeSet,0.35)



masterSet = [(neg, 0) for neg in negativeTrainSet]
masterSet.extend((neg, 0) for neg in negativeLantiTrainSet)
masterSet.extend((pos,1) for pos in positives)


n_epochs = 30
print_every = 100
frac_train = 0.95

n_hidden = 512

learning_rate = 0.001

model = RNN(20,n_hidden,2)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8,nesterov=True)

saved_model_path = 'model_peptide_SGD_epoch_58.ckpt'

if os.path.isfile(saved_model_path):
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("=> loaded checkpoint ")
    with open('logfile.log','a') as outfile:
        outfile.write("=> loaded checkpoint\n")

print('Testing on Entire Set')
print_every = 1

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

test_set_folder = '/Volumes/lab_data/Ripps/training_sets/test_sets'

with torch.no_grad():
    wrong_words = set()
    test_files = glob(os.path.join(test_set_folder,'*.txt'))
    #test_set = [(str(x.seq),x.name) for x in SeqIO.parse('/Users/emzodls/Downloads/GCA_001770815.1_ASM177081v1_orfs.fa','fasta')]
    test_set = []
    for x in SeqIO.parse('/Users/emzodls/Downloads/GCA_001770815.1_ASM177081v1_orfs.fa', 'fasta'):
        if all(y in amino_acids for y in str(x.seq)):
            test_set.append((str(x.seq),x.name))
    leader = 0
    not_leader = 0
    for seq,name in test_set:
        category, line, category_tensor, line_tensor = prepareTensors((seq,1))
        output = evaluate(model,line_tensor)
        guess = category_from_output(output)
        if guess[0] == 'word':
            leader += 1
        else:
            not_leader +=1
        print('Name = {}, Line = {}, Guess = {}'.format(name,line,guess))
    print('Total = {}, Leader = {}, Not Leader = {}'.format(len(test_set),leader,not_leader))
    #
    # wrong_words = set()
    # test_files = glob(os.path.join(test_set_folder,'*.txt'))
    # for test_set_file in test_files:
    #     test_pos = set()
    #     test_set_name = os.path.splitext(os.path.split(test_set_file)[1])[0]
    #     for line in open(test_set_file):
    #         line = line.strip().upper()
    #         if all(x in amino_acids for x in line):
    #             test_pos.add(line)
    #     print('Test Set: {}, Testing on {} Unique Percursor Peptides'.format(test_set_name,len(test_pos-positives)))
    #     correct_word = 0
    #     total_word = 0
    #     correct_not_word = 0
    #     total_not_word = 0
    #     total_val_loss = 0
    #     for idx,labeled_pair in enumerate((pos,1) for pos in test_pos-positives):
    #         category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
    #         output = evaluate(model,line_tensor)
    #         guess = category_from_output(output)
    #         if category == 'word':
    #             total_word += 1
    #             if guess[0] == category:
    #                 correct_word += 1
    #             else:
    #                 wrong_words.add(line)
    #         else:
    #             total_not_word += 1
    #             if guess[0] == category:
    #                 correct_not_word += 1
    #         if idx % print_every == 0:
    #             guess, guess_i = category_from_output(output)
    #             correct = '✓' if guess == category else '✗ (%s)' % category
    #             print('%d %d%% (%s) %s / %s %s' % (
    #                 idx, idx / len(positives) * 100, timeSince(start), line, guess, correct))
    #
    #     print('Correct Word = {} %'.format((correct_word / total_word) * 100))

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
