# melody-to-lyrics-transformer

How to use:
1. Install the requirements. You may need to adjust the tensorflow version according to your machine (Linux/Mac Intel/Mac M1...). We only use the Keras tokenizer in preprocess.py and don't need Tensorflow in training procedure.
2. Config your logger in pretrain.py before create pl.Trainer().
3. Run pretrain.py

You can change the data you want to use in Data folder.

