"""
@Project: cocogan_transformer
@File: transformer.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/08/23
"""


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from models import *
import torch.optim as optim
from criterion import get_adv_losses
from utils import *
from evaluation import get_bert_scores, get_rouge_scores, get_bleu_scores
import torch.nn.functional as F
from itertools import repeat


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class TransformerGenerator(pl.LightningModule):
    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            inner_dim,
            num_heads,
            num_layers,
            d_k,
            d_v,
            lyrics_vocab_size,
            pitch_size,
            duration_size,
            rest_size,
            max_seq_len,
            bos_idx_lyrics,
            pad_idx_lyrics,
            pad_idx_pitch,
            pad_idx_duration,
            pad_idx_rest,
            dropout,
            lr,
            tokenizer_lyrics_path=None,
            teacher_forcing=False,
            adding_noise=False
    ):

        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.generator = TransformerWithNoise(n_src_vocab=[self.hparams.pitch_size, self.hparams.duration_size, self.hparams.rest_size],
                                              n_trg_vocab=self.hparams.lyrics_vocab_size,
                                              src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                                              trg_pad_idx=self.hparams.pad_idx_lyrics,
                                              d_word_vec=self.hparams.embedding_dim,
                                              d_model=self.hparams.hidden_dim,
                                              d_inner=self.hparams.inner_dim,
                                              n_layers=self.hparams.num_layers,
                                              n_head=self.hparams.num_heads,
                                              d_k=self.hparams.d_k,
                                              d_v=self.hparams.d_v,
                                              dropout=self.hparams.dropout,
                                              n_position=self.hparams.max_seq_len,
                                              enc_dec_attn=self.hparams.teacher_forcing)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        # self.lr_scheduler = CosineWarmupScheduler(
        #     optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        # )
        return optimizer

    def _cal_loss(self, pred, target):
        pred_flat = pred.contiguous().view(-1, pred.size(-1))

        target_flat = target.contiguous().view(-1)

        loss = F.cross_entropy(pred_flat, target_flat)

        return loss

    def forward(self, seq_pitch, seq_duration, seq_rest, target, noise=None):
        if self.hparams.teacher_forcing:
            src_seq_pitch = seq_pitch[:, :-1]
            src_seq_duration = seq_duration[:, :-1]
            src_seq_rest = seq_rest[:, :-1]
            trg_seq = target[:, :-1]
            gold = target[:, 1:]
        else:
            src_seq_pitch = seq_pitch[:, 1:]
            src_seq_duration = seq_duration[:, 1:]
            src_seq_rest = seq_rest[:, 1:]
            trg_seq = target[:, 1:]
            gold = target[:, 1:]

        src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)
        trg_seq = trg_seq.to(device)

        assert src_seq_pitch.shape == src_seq_duration.shape == src_seq_rest.shape == trg_seq.shape == gold.shape

        pred = self.generator.forward(src_seq, trg_seq, noise)

        return pred, gold

    def training_step(self, batch, batch_idx):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target = batch[1]

        bs = target.shape[0]
        seq_len = self.hparams.max_seq_len

        if self.hparams.adding_noise:
            noise = torch.randn(bs, seq_len, self.hparams.hidden_dim).to(device)
        else:
            noise = None

        pred, gold = self.forward(seq_pitch, seq_duration, seq_rest, target, noise=noise)

        gen_samples = F.softmax(pred, dim=-1)
        real_samples = F.one_hot(gold, self.hparams.lyrics_vocab_size).float()

        if (self.current_epoch + 1) % 20 == 0:
            if batch_idx == 0:
                gt_lyrics = sequences_to_texts(real_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)
                self.logger.experiment["lyrics/train_gt"].log(gt_lyrics[0])
                # self.logger.experiment.log_text(gt_lyrics[0], step=self.current_epoch)

                gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)
                self.logger.experiment["lyrics/train_gen"].log(gen_lyrics[0])
                # self.logger.experiment.log_text(gen_lyrics[0], step=self.current_epoch)

        loss = self._cal_loss(pred, gold)

        self.log("pretrain_g_loss_train", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target = batch[1]
        bs = target.shape[0]

        if not self.hparams.teacher_forcing:
            pred, gold = self.forward(seq_pitch, seq_duration, seq_rest, target, noise=None)

        else:
            translator = Translator(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1, # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_lyrics,
                trg_bos_idx=self.hparams.bos_idx_lyrics,
                trg_eos_idx=self.hparams.pad_idx_lyrics).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc in zip(seq_pitch, seq_duration, seq_rest, target):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                seq_len = self.hparams.max_seq_len

                if self.hparams.adding_noise:
                    noise = torch.randn(1, seq_len, self.hparams.hidden_dim).to(device)
                else:
                    noise = None

                pred_seq = translator.translate_sentence(src_seq, noise=noise)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_lyrics).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)

            pred = torch.vstack(pred_list).to(device)
            gold = torch.vstack(gold_list).to(device)

        pred_np = pred[0].detach().cpu().numpy()
        gold_np = gold[0].detach().cpu().numpy()

        print('pred: ', pred_np)
        print('gold: ', gold_np)

        # gen_samples = F.softmax(pred, dim=-1)
        gen_samples = F.one_hot(pred, self.hparams.lyrics_vocab_size).float()
        real_samples = F.one_hot(gold, self.hparams.lyrics_vocab_size).float()

        gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)
        gt_lyrics = sequences_to_texts(real_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)

        if batch_idx == 0:
            if self.current_epoch == 0:
                # gt_lyrics = sequences_to_texts(real_samples)
                # self.logger.experiment.add_text('valid_gt', gt_lyrics[0], self.current_epoch)
                self.logger.experiment["lyrics/valid_gt"].log(gt_lyrics[0])
                # self.logger.experiment.log_text(gt_lyrics[0], step=self.current_epoch)

            if (self.current_epoch + 1) % 20 == 0:
                # gen_lyrics = sequences_to_texts(gen_samples)
                # self.logger.experiment.add_text('valid_gen', gen_lyrics[0], self.current_epoch)
                self.logger.experiment["lyrics/valid_gen"].log(gen_lyrics[0])
                # self.logger.experiment.log_text(gen_lyrics[0], step=self.current_epoch)

        gen_lyrics = remove_start_and_end(gen_lyrics)
        gt_lyrics = remove_start_and_end(gt_lyrics)

        rouge_1, rouge_2, rouge_l = get_rouge_scores(gt_lyrics, gen_lyrics)
        self.logger.experiment["val/rouge_1/precision"].log(rouge_1[0])
        self.logger.experiment["val/rouge_1/recall"].log(rouge_1[1])
        self.logger.experiment["val/rouge_1/f1"].log(rouge_1[2])
        self.logger.experiment["val/rouge_2/precision"].log(rouge_2[0])
        self.logger.experiment["val/rouge_2/recall"].log(rouge_2[1])
        self.logger.experiment["val/rouge_2/f1"].log(rouge_2[2])
        self.logger.experiment["val/rouge_l/precision"].log(rouge_l[0])
        self.logger.experiment["val/rouge_l/recall"].log(rouge_l[1])
        self.logger.experiment["val/rouge_l/f1"].log(rouge_l[2])

        bleu_2, bleu_3, bleu_4 = get_bleu_scores(gt_lyrics, gen_lyrics)
        self.logger.experiment["val/bleu/bleu_2"].log(bleu_2)
        self.logger.experiment["val/bleu/bleu_3"].log(bleu_3)
        self.logger.experiment["val/bleu/bleu_4"].log(bleu_4)

        # loss = self._cal_loss(pred, gold)
        gold_masked = torch.where(gold != self.hparams.pad_idx_lyrics, gold, -1).to(device)
        eq_count = (gold_masked == pred).sum()

        acc = eq_count / bs / self.hparams.max_seq_len

        # self.log("pretrain_g_loss_valid", loss)
        self.log("pretrain_g_acc_valid", acc)

        return acc

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target = batch[1]
        bs = target.shape[0]

        if not self.hparams.teacher_forcing:
            pred, gold = self.forward(seq_pitch, seq_duration, seq_rest, target, noise=None)

        else:
            translator = Translator(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1, # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_lyrics,
                trg_bos_idx=self.hparams.bos_idx_lyrics,
                trg_eos_idx=self.hparams.pad_idx_lyrics).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc in zip(seq_pitch, seq_duration, seq_rest, target):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                seq_len = self.hparams.max_seq_len

                if self.hparams.adding_noise:
                    noise = torch.randn(1, seq_len, self.hparams.hidden_dim).to(device)
                else:
                    noise = None

                pred_seq = translator.translate_sentence(src_seq, noise=noise)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_lyrics).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)

            pred = torch.vstack(pred_list).to(device)
            gold = torch.vstack(gold_list).to(device)

        pred_np = pred[0].detach().cpu().numpy()
        gold_np = gold[0].detach().cpu().numpy()

        print('pred: ', pred_np)
        print('gold: ', gold_np)

        # gen_samples = F.softmax(pred, dim=-1)
        gen_samples = F.one_hot(pred, self.hparams.lyrics_vocab_size).float()
        real_samples = F.one_hot(gold, self.hparams.lyrics_vocab_size).float()

        gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)
        gt_lyrics = sequences_to_texts(real_samples, tokenizer_lyr_path=self.hparams.tokenizer_lyrics_path)

        if batch_idx == 0:
            if self.current_epoch == 0:
                # gt_lyrics = sequences_to_texts(real_samples)
                # self.logger.experiment.add_text('valid_gt', gt_lyrics[0], self.current_epoch)
                self.logger.experiment["lyrics/valid_gt"].log(gt_lyrics[0])
                # self.logger.experiment.log_text(gt_lyrics[0], step=self.current_epoch)

            if (self.current_epoch + 1) % 20 == 0:
                # gen_lyrics = sequences_to_texts(gen_samples)
                # self.logger.experiment.add_text('valid_gen', gen_lyrics[0], self.current_epoch)
                self.logger.experiment["lyrics/valid_gen"].log(gen_lyrics[0])
                # self.logger.experiment.log_text(gen_lyrics[0], step=self.current_epoch)

        gen_lyrics = remove_start_and_end(gen_lyrics)
        gt_lyrics = remove_start_and_end(gt_lyrics)

        return gen_lyrics, gt_lyrics

