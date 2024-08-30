"""
@Project: CocoGAN
@File: evaluation.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/08/01
"""

import numpy as np

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.metrics.scores import (precision, recall)
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.functional.text.bert import bert_score

from utils import sequences_to_texts, remove_start_and_end

# samples = gen.sample(test_mels, 80, 32, start_letter=start_id_lyr)
# preds = tokenizer_lyr.sequences_to_texts(samples.cpu().detach().numpy())
# orig = tokenizer_lyr.sequences_to_texts(test_lyr)
#
# preds = remove_start_and_end(preds)
# orig = remove_start_and_end(orig)


def  get_rouge_scores(orig, preds):

    rouge = ROUGEScore()

    r_f_measure_1, r_precision_1, r_recall_1 = [], [], []
    r_f_measure_2, r_precision_2, r_recall_2 = [], [], []
    r_f_measure_l, r_precision_l, r_recall_l = [], [], []
    for test_ref, test_pred in zip(orig, preds):
        rouge_dict = rouge(test_pred, test_ref)
        rouge1_fmeasure = rouge_dict["rouge1_fmeasure"]
        rouge1_precision = rouge_dict["rouge1_precision"]
        rouge1_recall = rouge_dict["rouge1_recall"]
        rouge2_fmeasure = rouge_dict["rouge2_fmeasure"]
        rouge2_precision = rouge_dict["rouge2_precision"]
        rouge2_recall = rouge_dict["rouge2_recall"]
        rougeL_fmeasure = rouge_dict["rougeL_fmeasure"]
        rougeL_precision = rouge_dict["rougeL_precision"]
        rougeL_recall = rouge_dict["rougeL_recall"]

        r_f_measure_1.append(rouge1_fmeasure)
        r_precision_1.append(rouge1_precision)
        r_recall_1.append(rouge1_recall)
        r_f_measure_2.append(rouge2_fmeasure)
        r_precision_2.append(rouge2_precision)
        r_recall_2.append(rouge2_recall)
        r_f_measure_l.append(rougeL_fmeasure)
        r_precision_l.append(rougeL_precision)
        r_recall_l.append(rougeL_recall)

    # print(np.mean(r_f_measure_1), np.mean(r_precision_1), np.mean(r_recall_1))
    # print(np.mean(r_f_measure_2), np.mean(r_precision_2), np.mean(r_recall_2))
    # print(np.mean(r_f_measure_l), np.mean(r_precision_l), np.mean(r_recall_l))

    rouge_1_f1 = np.mean(r_f_measure_1)
    rouge_1_precision = np.mean(r_precision_1)
    rouge_1_recall = np.mean(r_recall_1)

    rouge_2_f1 = np.mean(r_f_measure_2)
    rouge_2_precision = np.mean(r_precision_2)
    rouge_2_recall = np.mean(r_recall_2)

    rouge_l_f1 = np.mean(r_f_measure_l)
    rouge_l_precision = np.mean(r_precision_l)
    rouge_l_recall = np.mean(r_recall_l)

    return [rouge_1_precision.round(4), rouge_1_recall.round(4), rouge_1_f1.round(4)], \
           [rouge_2_precision.round(4), rouge_2_recall.round(4), rouge_2_f1.round(4)], \
           [rouge_l_precision.round(4), rouge_l_recall.round(4), rouge_l_f1.round(4)]


def get_bleu_scores(orig, preds):
    chencherry = SmoothingFunction()

    bleus_4, bleus_3, bleus_2 = [], [], []
    for test_ref, test_pred in zip(orig, preds):
        test_ref = test_ref.split(' ')
        test_pred = test_pred.split(' ')
        bleu4 = sentence_bleu([test_ref], test_pred, smoothing_function=chencherry.method0)
        bleu3 = sentence_bleu([test_ref], test_pred, weights=[1 / 3, 1 / 3, 1 / 3], smoothing_function=chencherry.method0)
        bleu2 = sentence_bleu([test_ref], test_pred, weights=[1 / 2, 1 / 2], smoothing_function=chencherry.method0)
        bleus_4.append(bleu4)
        bleus_3.append(bleu3)
        bleus_2.append(bleu2)

    return np.mean(bleus_2).round(4), np.mean(bleus_3).round(4), np.mean(bleus_4).round(4)


def get_bert_scores(orig, preds):
    bertscore = BERTScore()
    scores = bertscore(orig, preds)
    # scores = bert_score(orig, preds)
    return [np.mean(scores["precision"]), np.mean(scores["recall"]), np.mean(scores["f1"])]

