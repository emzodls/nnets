## generate negative words for training, first get the words from all of the languages as a whitelist, then match the length distribution by drawing random lengths

## to do: try char_rnn_classification performance (training all but 1 language for word, 50% of negative words, try to classify other language with 5x negative words excluded from training

import os
from random import choice
from glob import glob
import matplotlib.pyplot as plt

os.chdir('/Volumes/lab_data/nnets')
os.chdir('/Volumes/lab_data/language_training')

files = glob("*.txt")
files
allWords = set()
for language in files:
    allWords.update(open(language).strip().split(','))

for language in files:
    allWords.update(open(language).read().strip().split(','))

word_length = [len(x) for x in allWords]
word_length
max(word_length)
min(word_length)
letters = set()
for word in allWords:
    for letter in word:
        letters.add(letter)

len(letters)

letters = list(letters)
letters
choice(letters)
choice(letters,6)
help(choice)
choice(word_length)
negatives = set()
while len(negatives) < len(allWords)*10:
    negLength = choice(word_length)
    negWord = ''.join(choice(letters) for _ in range(negLength))
    if negWord not in allWords:
        print(negWord)
        negatives.add(negWord)

neg_lengths = [len(x) for x in negatives]
plt.hist(neg_lengths)
plt.hist(word_length)
with open('negatives.txt','w') as outfile:
    for neg in negatives:
        outfile.write('{},'.format(neg))
