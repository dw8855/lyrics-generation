"""
@Project: cocogan_transformer
@File: hparams.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/08/24
"""

BATCH_SIZE = 64
MAX_LEN = 20

EMBEDDING_DIM = 256
EMBEDDING_DIM_ATTR = EMBEDDING_DIM / 4

HIDDEN_DIM = 256
INNER_DIM = 1024
D_K = 64
D_V = 64
NUM_HEADS = 3
NUM_LAYERS = 4
DROPOUT = 0.2
G_LR_PRE = 1e-4
G_LR_ADV = 1e-4

D_EMBEDDING_DIM = 64
D_FILTER_SIZES = [2, 3, 4, 5]
D_NUM_FILTERS = [300, 300, 300, 300]
D_NUM_REPS = 64
D_LR_ADV = 1e-4
