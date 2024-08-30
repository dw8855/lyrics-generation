"""
@Project: cocogan_transformer
@File: pretrain.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/08/24
"""


import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from transformer_generator import TransformerGenerator
from dataset import LyricsMelodyDataset, LyricsMelodyDataModule
from hparams import *

from pytorch_lightning.loggers import NeptuneLogger
import neptune.new as neptune


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
    data_folder = './data/syllable_and_word_' + str(MAX_LEN)
    tokenizer_folder = os.path.join(data_folder, 'tokenizers')

    lyrics_melody_dataset = LyricsMelodyDataset(data=os.path.join(data_folder, 'sequences.npy'),
                                                tokenizer_lyrics_path=os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'),
                                                tokenizer_pitch_path=os.path.join(tokenizer_folder, 'tokenizer_note.pkl'),
                                                tokenizer_duration_path=os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'),
                                                tokenizer_rest_path=os.path.join(tokenizer_folder, 'tokenizer_rest.pkl'))

    lyrics_melody_data_module = LyricsMelodyDataModule(data_train=os.path.join(data_folder, 'sequences_train.npy'),
                                                       data_val=os.path.join(data_folder, 'sequences_valid.npy'),
                                                       data_test=os.path.join(data_folder, 'sequences_test.npy'),
                                                       tokenizer_lyrics_path=os.path.join(tokenizer_folder,
                                                                                          'tokenizer_syllable.pkl'),
                                                       tokenizer_pitch_path=os.path.join(tokenizer_folder,
                                                                                         'tokenizer_note.pkl'),
                                                       tokenizer_duration_path=os.path.join(tokenizer_folder,
                                                                                            'tokenizer_duration.pkl'),
                                                       tokenizer_rest_path=os.path.join(tokenizer_folder,
                                                                                        'tokenizer_rest.pkl'),
                                                       batch_size=BATCH_SIZE)

    train_loader = lyrics_melody_data_module.train_dataloader()
    val_loader = lyrics_melody_data_module.val_dataloader()
    test_loader = lyrics_melody_data_module.test_dataloader()

    max_seq_len = MAX_LEN + 1

    vocab_size_lyrics, vocab_size_pitch, vocab_size_duration, vocab_size_rest = lyrics_melody_dataset.get_vocab_size()
    padding_idx_lyrics, padding_idx_pitch, padding_idx_duration, padding_idx_rest = lyrics_melody_dataset.get_padding_ids()
    start_id_lyrics, start_id_pitch, start_id_duration, start_id_rest = lyrics_melody_dataset.get_start_ids()

    root_dir = './checkpoints/pre'
    os.makedirs(root_dir, exist_ok=True)
    train_discriminator = False

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_gen_filename = os.path.join(CHECKPOINT_PATH, "test.ckpt")
    pretrained_gen_filename = 'checkpoints/pre/last-v1.ckpt'
    if os.path.isfile(pretrained_gen_filename):
        print("Found pretrained model, loading...")
        model_gen = TransformerGenerator.load_from_checkpoint(pretrained_gen_filename)
    else:
        # model = TransformerPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        # trainer.fit(model, train_loader, val_loader)

        # !!! You need to config a logger here. In default, I used a NeptuneLogger.
        # !!! You can also use other loggers or save values by your self.
        # !!! Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        neptune_logger = NeptuneLogger(
            project="NII-music-generation/VAE",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OGRiZjllMC1iYWZiLTQyM2QtYmFkNC00NTcwZjQ5OTRmYjgifQ==",
                # your credentials
            log_model_checkpoints=False,
            mode="debug",
        )

        trainer_gen = pl.Trainer(
            logger=neptune_logger,
            default_root_dir=root_dir,
            # callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            callbacks=[ModelCheckpoint(dirpath=root_dir, monitor='pretrain_g_loss_train', filename='pre-{epoch}-{step}', save_top_k=5, every_n_epochs=10, save_last=True)],
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=1000,
            gradient_clip_val=5,
            check_val_every_n_epoch=30
        )
        trainer_gen.logger._default_hp_metric = None  # Optional logging argument that we don't need

        model_gen = TransformerGenerator(embedding_dim=EMBEDDING_DIM,
                                         hidden_dim=HIDDEN_DIM,
                                         inner_dim=INNER_DIM,
                                         num_heads=NUM_HEADS,
                                         num_layers=NUM_LAYERS,
                                         d_k=D_K,
                                         d_v=D_V,
                                         lyrics_vocab_size=vocab_size_lyrics,
                                         pitch_size=vocab_size_pitch,
                                         duration_size=vocab_size_duration,
                                         rest_size=vocab_size_rest,
                                         max_seq_len=max_seq_len,
                                         bos_idx_lyrics=start_id_lyrics,
                                         pad_idx_lyrics=padding_idx_lyrics,
                                         pad_idx_pitch=padding_idx_pitch,
                                         pad_idx_duration=padding_idx_duration,
                                         pad_idx_rest=padding_idx_rest,
                                         dropout=DROPOUT,
                                         lr=G_LR_PRE,
                                         tokenizer_lyrics_path=os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'),
                                         teacher_forcing=True,
                                         adding_noise=None)

        trainer_gen.fit(model_gen, train_loader, val_loader)

