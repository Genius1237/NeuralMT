import numpy as np
import tensorflow as tf
import math
import pickle
import operator
import gensim
from gensim.models import word2vec
import logging

def readDataBible():
    '''
        Detects sentences based on the occurance of the full_stop_char_sans variable
        Inserts that between every 2 sentences as well
        As of now, removes occurances of ''
    '''
    filename = 'Bible/bible_translation.pickle'
    list_of_words = []
    words = {}
    word_count = 0
    full_stop_char_sans = 2404

    with open(filename, 'rb') as f:
        d = pickle.load(f)
    for spair in d:
        sans = spair[0]
        sans_words = []
        # sans.replace(ord(full_stop_char_sans),'')

        for w in sans.split(chr(full_stop_char_sans)):
            w1 = w.split(' ')
            sans_words.extend(w1)
            if len(w1) != 0 and w1[0] != '':
                sans_words.append(chr(full_stop_char_sans))

        for word in sans_words:
            '''
                Some preprocessing can be done here
                It will definitely help
            '''
            if word is '':
                continue
            if word not in words:
                words[word] = 1
                word_count += 1
            else:
                words[word] += 1
            list_of_words.append(word)

    return list_of_words

def main():
    '''
    list_of_words, words, word_count=readDataBible()

        list_of_words contains words in the order in which they appear. As it is
        word_count is the number of unique words in the vocabulary
        words is a dictionary. Key is word and value is the number of occurances of that word
    '''

    embedding_size = 64
    batch_size = 64
    window_size = 5

    print(readDataBible())
    input()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(readDataBible(),sg=1,size=embedding_size,window=window_size,workers=4)

    print(model.wv.index2word)
    
    print(model.wv.similarity('मम','तत्'))

if __name__ == "__main__":
    main()
