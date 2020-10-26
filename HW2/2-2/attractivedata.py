import pandas as pd
import torch
import torchtext
from torchtext import data
import nltk
import spacy
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class AttractiveData:
    def __init__(self, train_file, test_file, pretrained_file, max_size, min_freq):
        self.nlp_model = spacy.load('en_core_web_lg')
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        self.TEXT = data.Field(sequential=True, tokenize=self.tokenizer)
        self.LABEL = data.LabelField(dtype=torch.float, sequential=False)
        self.TEXT.build_vocab(self.df_train.Headline, vectors=torchtext.vocab.Vectors(pretrained_file), max_size=max_size, min_freq=min_freq)
        self.LABEL.build_vocab(self.df_train.Label)

    def tokenizer(self, corpus):
        return [token for token in self.nlp_model(corpus)]