"""
@Project: cocogan_transformer
@File: evaluate.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/11/15
"""


import numpy as np

from utils import sequences_to_texts, remove_start_and_end
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer_generator import TransformerGenerator
from dataset import LyricsMelodyDataset, LyricsMelodyDataModule
from hparams import *
from evaluation import *
import json

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

if __name__ == '__main__':
    # Path to the folder where the pretrained models are saved
    # CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")
    data_folder = './data/sentence_level_concat_20'
    tokenizer_folder = os.path.join(data_folder, 'tokenizers')

    lyrics_melody_dataset = LyricsMelodyDataset(data=os.path.join(data_folder, 'sequences.npy'),
                                                tokenizer_lyrics_path=os.path.join(tokenizer_folder,
                                                                                   'tokenizer_lyr.pkl'),
                                                tokenizer_pitch_path=os.path.join(tokenizer_folder,
                                                                                  'tokenizer_note.pkl'),
                                                tokenizer_duration_path=os.path.join(tokenizer_folder,
                                                                                     'tokenizer_duration.pkl'),
                                                tokenizer_rest_path=os.path.join(tokenizer_folder,
                                                                                 'tokenizer_rest.pkl'))

    lyrics_melody_data_module = LyricsMelodyDataModule(data_train=os.path.join(data_folder, 'sequences_train.npy'),
                                                       data_val=os.path.join(data_folder, 'sequences_valid.npy'),
                                                       data_test=os.path.join(data_folder, 'sequences_test.npy'),
                                                       tokenizer_lyrics_path=os.path.join(tokenizer_folder,
                                                                                          'tokenizer_lyr.pkl'),
                                                       tokenizer_pitch_path=os.path.join(tokenizer_folder,
                                                                                         'tokenizer_note.pkl'),
                                                       tokenizer_duration_path=os.path.join(tokenizer_folder,
                                                                                            'tokenizer_duration.pkl'),
                                                       tokenizer_rest_path=os.path.join(tokenizer_folder,
                                                                                        'tokenizer_rest.pkl'),
                                                       batch_size=BATCH_SIZE)

    test_loader = lyrics_melody_data_module.test_dataloader()

    vocab_size_lyrics, vocab_size_pitch, vocab_size_duration, vocab_size_rest = lyrics_melody_dataset.get_vocab_size()
    padding_id_lyrics, padding_id_pitch, padding_id_duration, padding_id_rest = lyrics_melody_dataset.get_padding_ids()
    start_id_lyrics, start_id_pitch, start_id_duration, start_id_rest = lyrics_melody_dataset.get_start_ids()

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_gen_filename = os.path.join(CHECKPOINT_PATH, "test.ckpt")
    pretrained_filename = 'checkpoints/pre/last-v1.ckpt'

    trainer_mle = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0)

    model_gan = TransformerGenerator.load_from_checkpoint(pretrained_filename)

    predictions = trainer_mle.predict(model_gan, dataloaders=test_loader)

    gen_lyrics = []
    gt_lyrics = []

    for batch in predictions:
        gen_lyrics = gen_lyrics + batch[0]
        gt_lyrics = gt_lyrics + batch[1]

    # with open("gen_lyrics.json", "w") as fp:
    #     json.dump(gen_lyrics, fp)
    #
    # with open("gt_lyrics.json", "w") as fp:
    #     json.dump(gt_lyrics, fp)

    rouge_1, rouge_2, rouge_l = get_rouge_scores(gt_lyrics, gen_lyrics)
    print(rouge_1, rouge_2, rouge_l)

    bleu_2, bleu_3, bleu_4 = get_bleu_scores(gt_lyrics, gen_lyrics)
    print(bleu_2, bleu_3, bleu_4)

    # bert_scores = get_bert_scores(gt_lyrics, gen_lyrics)
    # print(bert_scores)
