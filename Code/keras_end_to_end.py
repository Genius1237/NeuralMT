'''
- We start with input sequences from a domain (e.g. English sentences)
    and correspding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
'''

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from random import shuffle
import numpy as np
import pickle

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples=0
data_path=''

input_texts = []
target_texts = []
input_words = {}
target_words = {}

def process_fra_eng():
    data_path = 'fra.txt'
    num_samples = 50000
    with open('../Data/{}'.format(data_path), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    shuffle(lines)
    
    for line in lines[:min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')

        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        
        target_text = '\t' + ' ' + target_text + ' '+'\n'
        
        input_texts.append(input_text.split(' '))
        target_texts.append(target_text.split(' '))


        for word in input_text.split(' '):
            if word not in input_words:
                input_words[word]=len(input_words)
        
        for word in target_text.split(' '):
            if word not in target_words:
                target_words[word] = len(target_words)

def process_bible():
    data_path = 'bible_translation.pickle'
    with open('../Data/Bible/{}'.format(data_path),'rb') as f:
        p=pickle.load(f)
    
process_fra_eng()

num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
assert len(input_texts)==len(target_texts)
num_examples=len(input_texts)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


encoder_input_data = np.zeros((num_examples, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
decoder_input_data = np.zeros((num_examples, max_decoder_seq_length, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((num_examples, max_decoder_seq_length, num_decoder_tokens),dtype='float32')

for i in range(num_examples):
    for j in range(len(input_texts[i])):
        encoder_input_data[i,j,input_words[input_texts[i][j]]]=1
    for j in range(len(target_texts[i])):
        decoder_input_data[i,j,target_words[target_texts[i][j]]]=1
        if j>0:
            decoder_target_data[i, j-1, target_words[target_texts[i][j]]] = 1


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2)
# Save model
model.save('keras_seq2seq_{}_{}_{}_{}_{}.h5'.format(data_path,batch_size,epochs,latent_dim,num_samples))


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict((i, word) for word, i in input_words.items())
reverse_target_word_index = dict((i, word) for word, i in target_words.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_words['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_word_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index:seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
