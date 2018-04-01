import numpy as np
import tensorflow as tf
import math
import pickle
import operator
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

def readDataBible():
    '''
        Detects sentences based on the occurance of the full_stop_char_sans variable
        Inserts that between every 2 sentences as well
        As of now, removes occurances of ''
    '''
    filename = 'Bible/bible_translation.pickle'
    list_of_words=[]
    words={}
    word_count=0
    full_stop_char_sans=2404
    
    with open(filename,'rb') as f:
        d=pickle.load(f)
    for spair in d:
        sans=spair[0]
        sans_words=[]
        #sans.replace(ord(full_stop_char_sans),'')

        for w in sans.split(chr(full_stop_char_sans)):
            sans_words.extend(w.split(' '))
            sans_words.append(chr(full_stop_char_sans))
        
        for word in sans_words:
            '''
                Some preprocessing can be done here
                It will definitely help
            '''
            if word is '':
                continue
            if word not in words:
                words[word]=1
                word_count+=1
            else:
                words[word]+=1
                list_of_words.append(word)

    return list_of_words,words,word_count


def main():
    list_of_words, words, word_count=readDataBible()
    '''
        list_of_words contains words in the order in which they appear. As it is
        word_count is the number of unique words in the vocabulary
        words is a dictionary. Key is word and value is the number of occurances of that word
    '''

    embedding_size = 64
    batch_size = 64
    window_size=5

    vocabulary_size = word_count
    dec_word_list = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    dic={}
    for word in dec_word_list:
        dic[word[0]]=word[1]

    reverse_dic={}
    for word in dic:
        reverse_dic[dic[word]]=word
    
    '''
    print(vocabulary_size)
    print(dec_word_list)
    print(dic)
    print(reverse_dic)
    '''
    for i in range(len(list_of_words)):
        list_of_words[i]=dic[list_of_words[i]]

    sampling_table = sequence.make_sampling_table(vocabulary_size)
    couples, labels = skipgrams(list_of_words, vocabulary_size, window_size=window_size, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")

    print(couples[:10], labels[:10])

    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(vocabulary_size, embedding_size, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((embedding_size, 1))(target)
    context = embedding(input_context)
    context = Reshape((embedding_size, 1))(context)

    # now perform the dot product operation to get a similarity measure
    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # setup a cosine similarity operation which will be output in a secondary model
    similarity = merge([target, context], mode='cos', dot_axes=0)
    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(input=[input_target, input_context], output=similarity)


if __name__=="__main__":
    main()
