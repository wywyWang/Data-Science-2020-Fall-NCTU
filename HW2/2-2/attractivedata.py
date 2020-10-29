import pandas as pd
import torch
import torchtext
from torchtext import data
import nltk
import spacy


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

        # self.df_test[['ID', 'Headline', 'Category']].to_csv('./data/new_test.csv', index=False)

        self.TEXT = data.Field(sequential=True, init_token='<s>', lower=False, tokenize=self.tokenizer, fix_length=max_size, pad_token='0')
        self.LABEL = data.Field(dtype=torch.float, sequential=False, use_vocab=False)
        self.CATEGORIES_LABEL = data.LabelField(sequential=False)
        # self.CATEGORIES_LABEL = data.Field(sequential=False, use_vocab=False)

        # self.train_data, self.test_data = data.TabularDataset.splits(
        #     path=path, train=train_filename, test=test_filename, format="csv", skip_header=True, 
        #     fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        # )

        self.train_data = data.TabularDataset(
            path=train_file, format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        )
        self.test_data = data.TabularDataset(
            path='./data/new_test.csv', format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', None)]
        )

        self.TEXT.build_vocab(self.train_data, vectors=pretrained_file, min_freq=min_freq)
        self.LABEL.build_vocab(self.train_data)
        self.CATEGORIES_LABEL.build_vocab(self.train_data)

        self.trainloader = data.BucketIterator(self.train_data, sort_key=lambda x: len(x.Text), batch_size=batch_size, device=self.device, train=True)

    def tokenizer(self, corpus):
        return [str(token) for token in self.nlp_model(corpus)]