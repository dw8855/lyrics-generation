"""
@Project: cocogan_transformer
@File: random_baseline.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/11/25
"""

from utils import load_tokenizer, remove_start_and_end
import numpy as np
import random
from evaluation import get_bert_scores, get_rouge_scores, get_bleu_scores
import os

data_folder = './data/sentence_level_concat_100'
tokenizer_folder = os.path.join(data_folder, 'tokenizers')
tokenizer_lyrics_path = os.path.join(tokenizer_folder, 'tokenizer_lyr.pkl')
data_test = os.path.join(data_folder, 'sequences_test.npy')

tokenizer_lyrics = load_tokenizer(tokenizer_lyrics_path)

syllable_counts = dict(sorted(tokenizer_lyrics.word_counts.items(), key=lambda item: item[1], reverse=True))
del syllable_counts['<bos>']
del syllable_counts['<eos>']
syllable_dict = tokenizer_lyrics.word_index
del syllable_dict['<unk>']
del syllable_dict['<bos>']
del syllable_dict['<eos>']

assert syllable_counts.keys() == syllable_dict.keys()

total_tokens = sum(syllable_counts.values())

token_lists = list(syllable_dict.values())
probability_list = [ x / total_tokens for x in list(syllable_counts.values())]

test_data = np.load(data_test)
test_tokens = test_data[:, :, 0]

num_samples = len(test_data)
length = 100

gen_samples_uniform = []
gen_samples_distribution = []

for i in range(num_samples):
    seq_sample_uniform = random.choices(token_lists, k=length)
    gen_samples_uniform.append(seq_sample_uniform)
    seq_sample_distribution = random.choices(token_lists, weights=probability_list, k=length)
    gen_samples_distribution.append(seq_sample_distribution)

test_lyrics = tokenizer_lyrics.sequences_to_texts(test_tokens)
test_lyrics = remove_start_and_end(test_lyrics)

# matching lengths
for i in range(num_samples):
    length_test = len(test_lyrics[i].split(' '))
    gen_samples_uniform[i] = gen_samples_uniform[i][:length_test]
    gen_samples_distribution[i] = gen_samples_distribution[i][:length_test]

gen_lyrics_uniform = tokenizer_lyrics.sequences_to_texts(gen_samples_uniform)
gen_lyrics_distribution = tokenizer_lyrics.sequences_to_texts(gen_samples_distribution)


rouge_1_uni, rouge_2_uni, rouge_l_uni = get_rouge_scores(test_lyrics, gen_lyrics_uniform)
bleu_2_uni, bleu_3_uni, bleu_4_uni = get_bleu_scores(test_lyrics, gen_lyrics_uniform)

rouge_1_dst, rouge_2_dst, rouge_l_dst = get_rouge_scores(test_lyrics, gen_lyrics_distribution)
bleu_2_dst, bleu_3_dst, bleu_4_dst = get_bleu_scores(test_lyrics, gen_lyrics_distribution)


print('Sampled from uniform distribution: ')
print('rouge 1/2/l: \n', rouge_1_uni, rouge_2_uni, rouge_l_uni)
print('bleu 2/3/4: \n', bleu_2_uni, bleu_3_uni, bleu_4_uni)
print('Sampled from data distribution: ')
print('rouge 1/2/l: \n', rouge_1_dst, rouge_2_dst, rouge_l_dst)
print('bleu 2/3/4: \n', bleu_2_dst, bleu_3_dst, bleu_4_dst)
