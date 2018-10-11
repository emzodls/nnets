import re
from unidecode import unidecode
from glob import glob
import os

data_source = '/Volumes/lab_data/language_training/it'

def process(text,min_size):
    words = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = unidecode(words)
    words = words.lower()
    words = words.split()

    return set(x for x in words if len(x) >= min_size)

def process_lang(path):
    dictionary = set()
    files_to_process = glob(os.path.join(path,'*.txt'))
    for document in files_to_process:
        dictionary.update(process(open(document, encoding="utf-8").read(),4))
    return dictionary


test = process_lang(data_source)
with open('/Volumes/lab_data/language_training/italian.txt','w') as outfile:
    outfile.write(','.join(test))