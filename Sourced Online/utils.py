import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import pickle as pickle
import numpy as np

def load_data():
    train_data_path = './data/train.p'
    val_data_path = './data/val.p'
    reverse_dictionary_path = './data/reverse_dictionary.p'

    train_data = pickle.load(open(train_data_path, 'rb'))
    print("Loaded train data!")
    val_data = pickle.load(open(val_data_path, 'rb'))
    print("Loaded val data!")
    reverse_dictionary = pickle.load(open(reverse_dictionary_path, 'rb'))
    print("Loaded reverse dictionary!")
    return train_data, val_data, reverse_dictionary

def print_closest_words(val_index, nearest, reverse_dictionary):
    val_word = reverse_dictionary[val_index]                 
    log_str = "Nearest to %s:" % val_word                          
    for k in range(len(nearest)):                                        
        close_word = reverse_dictionary[nearest[k]]                
        log_str = "%s %s," % (log_str, close_word)                 
    print(log_str)