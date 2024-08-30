"""
@Project: CocoGAN
@File: criterion.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/07/31
"""

import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.metrics.scores import (precision, recall)
from utils import load_tokenizer, remove_start_and_end


def get_adv_losses(d_out_real, d_out_fake, gen_samples=None, real_samples=None, loss_type='JS',
                   tokenizer_lyr_path='./tokenizers/tokenizer_lyr.pkl'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()
    tokenizer_lyr = load_tokenizer(tokenizer_lyr_path)

    if loss_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

    elif loss_type == 'JS':  # the vanilla GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = torch.mean(-d_out_fake)

    elif loss_type == 'hinge':  # the hinge loss
        d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
        d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -torch.mean(d_out_fake)

    elif loss_type == 'tv':  # the total variation distance
        d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
        g_loss = torch.mean(-nn.Tanh(d_out_fake))

    elif loss_type == 'rsgan':  # relativistic standard GAN
        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

    # if gen_samples == None or real_samples == None:
    #     return g_loss, d_loss
    #
    # r_s = torch.argmax(real_samples, dim=2)
    # r_s = tokenizer_lyr.sequences_to_texts(r_s.cpu().numpy())
    # g_s = torch.argmax(gen_samples, dim=2)
    # g_s = tokenizer_lyr.sequences_to_texts(g_s.cpu().numpy())
    #
    # r_s = remove_start_and_end(r_s)
    # g_s = remove_start_and_end(g_s)
    #
    # chencherry = SmoothingFunction()
    # bleus_4, precisions, recalls = [], [], []
    # try:
    #     for test_ref, test_pred in zip(r_s, g_s):
    #         bleu4 = sentence_bleu(test_ref, test_pred, smoothing_function=chencherry.method1)
    #         bleus_4.append(bleu4)
    #         test_ref_set = set(test_ref.split())
    #         test_pred_set = set(test_pred.split())
    #         prec = precision(test_ref_set, test_pred_set)
    #         rec = recall(test_ref_set, test_pred_set)
    #         prec = 1e-10 if (prec == None) else prec
    #         rec = 1e-10 if (rec == None) else rec
    #
    #         precisions.append(prec)
    #         recalls.append(rec)
    #
    #     bleus_4 = torch.tensor(bleus_4)
    #     bleu = torch.mean(bleus_4)
    #
    #     precisions = torch.tensor(precisions)
    #     precision_mean = torch.mean(precisions)
    #
    #     recalls = torch.tensor(recalls)
    #     recall_mean = torch.mean(recalls)
    #
    #     g_loss += -torch.log(bleu) - torch.log(precision_mean) - torch.log(recall_mean)
    #
    # except KeyError:
    #     g_loss += torch.tensor(20.0)
    #     return g_loss, d_loss

    return g_loss, d_loss
