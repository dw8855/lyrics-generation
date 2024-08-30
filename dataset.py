"""
@Project: CocoGAN
@File: dataset.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/07/19
"""

from torch.utils.data import Dataset, DataLoader
import random
import torch
from utils import load_tokenizer
import pytorch_lightning as pl
from typing import Optional
import numpy as np


class LyricsMelodyDataset(Dataset):

    def __init__(self, data, tokenizer_lyrics_path, tokenizer_pitch_path, tokenizer_duration_path, tokenizer_rest_path):
        self.data = np.load(data)
        self.tokenizer_lyrics = load_tokenizer(tokenizer_lyrics_path)
        self.tokenizer_pitch = load_tokenizer(tokenizer_pitch_path)
        self.tokenizer_duration = load_tokenizer(tokenizer_duration_path)
        self.tokenizer_rest = load_tokenizer(tokenizer_rest_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lyrics = self.data[idx, :, 0]
        pitch = self.data[idx, :, 1]
        duration = self.data[idx, :, 2]
        rest = self.data[idx, :, 3]

        # inp = torch.LongTensor(lyrics[:-1])
        # condition_pitch = torch.LongTensor(pitch[:-1]).unsqueeze(1)
        # condition_duration = torch.LongTensor(duration[:-1]).unsqueeze(1)
        # condition_rest = torch.LongTensor(rest[:-1]).unsqueeze(1)
        #
        # condition = torch.cat([condition_pitch, condition_duration, condition_rest], dim=1)
        #
        # target = torch.LongTensor(lyrics[1:])

        return (torch.LongTensor(pitch), torch.LongTensor(duration), torch.LongTensor(rest)), torch.LongTensor(lyrics)

    def get_vocab_size(self):
        # vocab_size_lyrics = min(len(self.tokenizer_lyrics.word_index) + 1, 10000)
        vocab_size_lyrics = len(self.tokenizer_lyrics.word_index) + 1
        vocab_size_pitch = len(self.tokenizer_pitch.word_index) + 1
        vocab_size_duration = len(self.tokenizer_duration.word_index) + 1
        vocab_size_rest = len(self.tokenizer_rest.word_index) + 1

        return vocab_size_lyrics, vocab_size_pitch, vocab_size_duration, vocab_size_rest

    def get_padding_ids(self, padding_id="<eos>"):
        # padding_id_lyrics = self.tokenizer_lyrics.word_index["eos"]
        # padding_id_pitch = self.tokenizer_pitch.word_index["eos"]
        # padding_id_duration = self.tokenizer_duration.word_index["eos"]
        # padding_id_rest = self.tokenizer_rest.word_index["eos"]
        padding_id_lyrics = self.tokenizer_lyrics.word_index[padding_id]
        padding_id_pitch = self.tokenizer_pitch.word_index[padding_id]
        padding_id_duration = self.tokenizer_duration.word_index[padding_id]
        padding_id_rest = self.tokenizer_rest.word_index[padding_id]

        return padding_id_lyrics, padding_id_pitch, padding_id_duration, padding_id_rest

    def get_start_ids(self, start_id="<bos>"):
        # start_id_lyrics = self.tokenizer_lyrics.word_index["bos"]
        # start_id_pitch = self.tokenizer_pitch.word_index["bos"]
        # start_id_duration = self.tokenizer_duration.word_index["bos"]
        # start_id_rest = self.tokenizer_rest.word_index["bos"]
        start_id_lyrics = self.tokenizer_lyrics.word_index[start_id]
        start_id_pitch = self.tokenizer_pitch.word_index[start_id]
        start_id_duration = self.tokenizer_duration.word_index[start_id]
        start_id_rest = self.tokenizer_rest.word_index[start_id]

        return start_id_lyrics, start_id_pitch, start_id_duration, start_id_rest

    def get_seq_len(self):
        return


class LyricsMelodyDataModule(pl.LightningDataModule):

    def __init__(self, data_train, data_val, data_test,
                 tokenizer_lyrics_path, tokenizer_pitch_path, tokenizer_duration_path, tokenizer_rest_path,
                 batch_size, add_token=False):
        super().__init__()
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.tokenizer_lyrics_path = tokenizer_lyrics_path
        self.tokenizer_pitch_path = tokenizer_pitch_path
        self.tokenizer_duration_path = tokenizer_duration_path
        self.tokenizer_rest_path = tokenizer_rest_path
        self.batch_size = batch_size
        self.add_token = add_token

        self.train_dataset = LyricsMelodyDataset(data=self.data_train,
                                                 tokenizer_lyrics_path=self.tokenizer_lyrics_path,
                                                 tokenizer_pitch_path=self.tokenizer_pitch_path,
                                                 tokenizer_duration_path=self.tokenizer_duration_path,
                                                 tokenizer_rest_path=self.tokenizer_rest_path)
        self.val_dataset = LyricsMelodyDataset(data=self.data_val,
                                               tokenizer_lyrics_path=self.tokenizer_lyrics_path,
                                               tokenizer_pitch_path=self.tokenizer_pitch_path,
                                               tokenizer_duration_path=self.tokenizer_duration_path,
                                               tokenizer_rest_path=self.tokenizer_rest_path)
        self.test_dataset = LyricsMelodyDataset(data=self.data_test,
                                                tokenizer_lyrics_path=self.tokenizer_lyrics_path,
                                                tokenizer_pitch_path=self.tokenizer_pitch_path,
                                                tokenizer_duration_path=self.tokenizer_duration_path,
                                                tokenizer_rest_path=self.tokenizer_rest_path)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)


if __name__ == '__main__':
    dataset = LyricsMelodyDataset(data='./data/sequences.npy',
                                  tokenizer_lyrics_path='./tokenizers/tokenizer_lyr.pkl',
                                  tokenizer_pitch_path='./tokenizers/tokenizer_note.pkl',
                                  tokenizer_duration_path='./tokenizers/tokenizer_duration.pkl',
                                  tokenizer_rest_path='./tokenizers/tokenizer_rest.pkl')

    print(len(dataset))
    vocab_size_lyrics, vocab_size_pitch, vocab_size_duration, vocab_size_rest = dataset.get_vocab_size()
    padding_id_lyrics, padding_id_pitch, padding_id_duration, padding_id_rest = dataset.get_padding_ids()

    data_module = LyricsMelodyDataModule(data_train='./data/sequences_train.npy',
                                         data_val='./data/sequences_test.npy',
                                         data_test='./data/sequences_test.npy',
                                         tokenizer_lyrics_path='./tokenizers/tokenizer_lyr.pkl',
                                         tokenizer_pitch_path='./tokenizers/tokenizer_note.pkl',
                                         tokenizer_duration_path='./tokenizers/tokenizer_duration.pkl',
                                         tokenizer_rest_path='./tokenizers/tokenizer_rest.pkl',
                                         batch_size=32)


    for batch in data_module.train_dataloader():
        print(len(batch))
        inp, condition = batch[0]
        target = batch[1]
        break