import pandas as pd
import torch
import torchtext
from torchtext import data
import nltk
import spacy
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class AttractiveData:
    def __init__(self, train_file, test_file, pretrained_file, max_size, min_freq, batch_size):
        path = train_file.split('/')[0] + '/'
        train_filename = train_file.split('/')[1]
        test_filename = test_file.split('/')[1]

        self.max_size = max_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nlp_model = spacy.load('en_core_web_lg')
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        self.TEXT = data.Field(sequential=True, init_token='<s>', tokenize=self.tokenizer, fix_length=max_size, pad_token='0')
        self.LABEL = data.LabelField(dtype=torch.float, sequential=False)

        self.train_data, self.test_data = data.TabularDataset.splits(
            path=path, train=train_filename, test=test_filename, format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', None), ('Label', self.LABEL)]
        )

        self.TEXT.build_vocab(self.train_data, vectors=torchtext.vocab.Vectors(pretrained_file), min_freq=min_freq)
        self.LABEL.build_vocab(self.train_data)

        self.trainloader, self.testloader = data.BucketIterator.splits(
            (self.train_data, self.test_data), sort_key=lambda x: len(x.Text), batch_size=batch_size, device=self.device)

        # self.padding_train = self.padding(self.df_train.Headline.to_list())
        # self.padding_test = self.padding(self.df_test.Headline.to_list())

    def tokenizer(self, corpus):
        return [str(token) for token in self.nlp_model(corpus)]

    def padding(self, data_seq):
        padding_seq = []
        for index in range(len(data_seq)):
            each_seq = self.tokenizer(data_seq[index]).copy()
            if len(data_seq[index]) < self.max_size:
                each_seq += ['0'] * (self.max_size - len(data_seq[index]))
                padding_seq.append(each_seq)
            else:
                each_seq = each_seq[:self.max_size]
                padding_seq.append(each_seq)
        return torch.tensor(padding_seq)