"""
@Project: cocogan_transformer
@File: preprocess_multi_level.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/12/29
"""


import os

import keras_preprocessing.text
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
# https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2/71838765#71838765
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.api._v2.keras.preprocessing.text import Tokenizer
# from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

songs_path = './data/Sentence_and_Word_Parsing'
# songs_path = './data/lmd-full_MIDI_dataset/Syllable_Parsing'
songs = os.listdir(songs_path)

# Done: Concatenate sentence level sequences to construct target-length sequences
# TODO: Data augmentation with overlapped slices

# the target length for dataset
max_len = 20

features = []
for song in tqdm(songs):
    features.append(np.load(f"{songs_path}/{song}", allow_pickle=True))

melody = np.array(features)[:, :, 1]
lyrics_word = np.array(features)[:, :, 2]
lyrics_syllable = np.array(features)[:, :, 3]

# Done: Add symbols to bos and eos tokens

lyrics_word_list = []
lyrics_syllable_list = []
lyrics_length_list = []

# Done: Also process word sequences

notes_list, durations_list, rests_list = [], [], []
melody_length_list = []

song_phrase_length = []
# read the data from original dataset
for song_syllable in tqdm(lyrics_syllable):
    phrase_length = []
    for phrase_syllable in song_syllable[0]:
        phrase_syllable = sum(phrase_syllable, [])

        phrase_length.append(len(phrase_syllable))

    song_phrase_length.append(phrase_length)

# concatenate the sentence-level sequences until reaching the target length
for song_word, song_syllable, song_melody in tqdm(zip(lyrics_word, lyrics_syllable, melody)):
    word_list = []
    syllable_list = []
    melody_list = []
    for idx, (phrase_word, phrase_syllable, phrase_melody) in \
            enumerate(zip(song_word[0], song_syllable[0], song_melody[0])):

        assert len(phrase_word) == len(phrase_syllable) == len(phrase_melody)
        # bad data with blank
        if not phrase_word:
            continue

        phrase_word_flat = sum(phrase_word, [])
        phrase_syllable_flat = sum(phrase_syllable, [])
        phrase_melody_flat = [n for word_level in phrase_melody for n in word_level]

        # concatenate the sentences
        if len(word_list) + len(phrase_word_flat) <= max_len:
            word_list.extend(phrase_word_flat)
            syllable_list.extend(phrase_syllable_flat)
            melody_list.extend(phrase_melody_flat)
            if idx == len(song_word[0]) - 1:
                if len(word_list) < 2:
                    continue
                # save
                assert len(word_list) == len(syllable_list) == len(melody_list)
                words = "<bos> " + " ".join(word_list) + " <eos>"
                syllables = "<bos> " + " ".join(syllable_list) + " <eos>"

                note = np.array(melody_list)[:, 0]
                duration = np.array(melody_list)[:, 1]
                rest = np.array(melody_list)[:, 2]
                note_str = "<bos> " + " ".join([str(n) for n in note]) + " <eos>"
                duration_str = "<bos> " + " ".join([str(d) for d in duration]) + " <eos>"
                rest_str = "<bos> " + " ".join([str(r) for r in rest]) + " <eos>"

                lyrics_word_list.append(words)
                lyrics_syllable_list.append(syllables)
                notes_list.append(note_str)
                durations_list.append(duration_str)
                rests_list.append(rest_str)
                lyrics_length_list.append(len(syllable_list))

                # initialize
                word_list = []
                syllable_list = []
                melody_list = []
        else: # reaching the target length
            # save
            if not word_list:
                word_list.extend(phrase_word_flat)
                syllable_list.extend(phrase_syllable_flat)
                melody_list.extend(phrase_melody_flat)
            # filter some bad weird long data
            if len(word_list) > max_len:
                # word_list = word_list[:max_len]
                # syllable_list = syllable_list[:max_len]
                # melody_list = melody_list[:max_len]
                continue
            # filter some bad weird short data
            if len(word_list) < 2:
                continue

            assert len(word_list) == len(syllable_list) == len(melody_list)

            words = "<bos> " + " ".join(word_list) + " <eos>"
            syllables = "<bos> " + " ".join(syllable_list) + " <eos>"

            note = np.array(melody_list)[:, 0]
            duration = np.array(melody_list)[:, 1]
            rest = np.array(melody_list)[:, 2]
            note_str = "<bos> " + " ".join([str(n) for n in note]) + " <eos>"
            duration_str = "<bos> " + " ".join([str(d) for d in duration]) + " <eos>"
            rest_str = "<bos> " + " ".join([str(r) for r in rest]) + " <eos>"

            lyrics_word_list.append(words)
            lyrics_syllable_list.append(syllables)
            notes_list.append(note_str)
            durations_list.append(duration_str)
            rests_list.append(rest_str)
            lyrics_length_list.append(len(syllable_list))

            # initialize
            # word_list = []
            # syllable_list = []
            # melody_list = []
            # start from the current phrase
            # this is a fix of previous data split
            word_list = phrase_word_flat
            syllable_list = phrase_syllable_flat
            melody_list = phrase_melody_flat

# Building tokenizers
# TODO: replacing the infrequent tokens by a specific one
tokenizer_syllable = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_syllable.fit_on_texts(lyrics_syllable_list)

tokenizer_word = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_word.fit_on_texts(lyrics_word_list)

print('Syllable count: ', tokenizer_syllable.word_counts)
print('Word count: ', tokenizer_word.word_counts)

pad_id_syllable = tokenizer_syllable.word_index["<eos>"]
start_id_syllable = tokenizer_syllable.word_index["<bos>"]

sequences_syllable_tokens = tokenizer_syllable.texts_to_sequences(lyrics_syllable_list)
sequences_syllable_tokens = pad_sequences(sequences_syllable_tokens,
                                          maxlen=max_len+2, truncating='post', padding='post', value=pad_id_syllable)

pad_id_word = tokenizer_word.word_index["<eos>"]
start_id_word = tokenizer_word.word_index["<bos>"]

sequences_word_tokens = tokenizer_word.texts_to_sequences(lyrics_word_list)
sequences_word_tokens = pad_sequences(sequences_word_tokens,
                                      maxlen=max_len + 2, truncating='post', padding='post', value=pad_id_word)

# create note sequences
tokenizer_note = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_note.fit_on_texts(notes_list)

pad_id_note = tokenizer_note.word_index["<eos>"]
start_id_note = tokenizer_note.word_index["<bos>"]

sequences_note = tokenizer_note.texts_to_sequences(notes_list)
sequences_note = pad_sequences(sequences_note,
                               truncating='post', padding='post', value=pad_id_note)

# create duration sequences
tokenizer_duration = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_duration.fit_on_texts(durations_list)

pad_id_duration = tokenizer_duration.word_index["<eos>"]
start_id_duration = tokenizer_duration.word_index["<bos>"]

sequences_duration = tokenizer_duration.texts_to_sequences(durations_list)
sequences_duration = pad_sequences(sequences_duration,
                                   truncating='post', padding='post', value=pad_id_duration)

# create rest sequences
tokenizer_rest = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_rest.fit_on_texts(rests_list)

pad_id_rest = tokenizer_rest.word_index["<eos>"]
start_id_rest = tokenizer_rest.word_index["<bos>"]

sequences_rest = tokenizer_rest.texts_to_sequences(rests_list)
sequences_rest = pad_sequences(sequences_rest,
                               truncating='post', padding='post', value=pad_id_rest)

sequences_syllable = np.expand_dims(sequences_syllable_tokens, axis=2)
sequences_word = np.expand_dims(sequences_word_tokens, axis=2)

sequences_note = np.expand_dims(sequences_note, axis=2)
sequences_duration = np.expand_dims(sequences_duration, axis=2)
sequences_rest = np.expand_dims(sequences_rest, axis=2)

# If only syllables are required, modify this line
sequences = np.concatenate([sequences_syllable, sequences_word, sequences_note, sequences_duration, sequences_rest], axis=2)

sequences, duplicated_idx = np.unique(sequences, axis=0, return_index=True)
sequences_unique = np.unique(sequences, axis=0)

assert len(sequences_unique) == len(sequences)

# saving dataset
data_folder = './data/syllable_and_word_' + str(max_len)
tokenizer_folder = os.path.join(data_folder, 'tokenizers')

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

if not os.path.exists(tokenizer_folder):
    os.mkdir(tokenizer_folder)

# a note for loading tokenizers
# def load_tokenizer(file):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         tokenizer = data['tokenizer']
#     return tokenizer

def save_tokenizer(file, tokenizer):
    with open(file, 'wb') as handle:
        pickle.dump({'tokenizer': tokenizer}, handle)

save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'), tokenizer_syllable)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_word.pkl'), tokenizer_word)

save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_note.pkl'), tokenizer_note)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'), tokenizer_duration)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_rest.pkl'), tokenizer_rest)

# Done: Add validation set
num_sequences = len(sequences)
val_test_len = int(num_sequences * 0.1) * 2

val_test_inds = np.random.choice(np.arange(num_sequences), size=val_test_len, replace=False)
train_inds = np.delete(np.arange(num_sequences), val_test_inds)
valid_inds = np.random.choice(val_test_inds, size=int(val_test_len/2), replace=False)
test_inds = np.delete(val_test_inds, list(np.where(val_test_inds == idx) for idx in valid_inds))

sequences_train = sequences[train_inds]
sequences_valid = sequences[valid_inds]
sequences_test = sequences[test_inds]

print('length of train, valid, and test data: ', len(sequences_train), len(sequences_valid), len(sequences_test))

np.save(os.path.join(data_folder, 'sequences.npy'), sequences)
np.save(os.path.join(data_folder, 'sequences_train.npy'), sequences_train)
np.save(os.path.join(data_folder, 'sequences_valid.npy'), sequences_valid)
np.save(os.path.join(data_folder, 'sequences_test.npy'), sequences_test)
