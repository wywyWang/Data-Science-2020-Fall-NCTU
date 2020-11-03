import pandas as pd
import torch
import torchtext
from torchtext import data


class AttractiveData:
    def __init__(self, train_file, val_file, test_file, pretrained_file, config):
        self.config = config
        self.preprocess(train_file, val_file, test_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df_train = pd.read_csv('./example/new_train.csv')
        self.df_val = pd.read_csv('./example/new_val.csv')
        self.df_test = pd.read_csv('./data/new_test.csv')

        # self.TEXT = data.Field(sequential=True, init_token='<s>', lower=False, tokenize=self.tokenizer, fix_length=max_size, pad_token='0')
        # self.TEXT = data.Field(sequential=True, lower=False, tokenize=self.tokenizer, fix_length=self.config['max_size'], pad_token='0')
        self.TEXT = data.Field(sequential=True, lower=True, batch_first=True)
        self.CATEGORIES_LABEL = data.LabelField(sequential=False, batch_first=True)
        self.LABEL = data.Field(dtype=torch.float, sequential=False, use_vocab=False, batch_first=True)
        self.ID = data.Field(sequential=False, use_vocab=False, batch_first=True)

        self.train_data = data.TabularDataset(
            path='./example/new_train.csv', format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        )
        self.val_data = data.TabularDataset(
            path='./example/new_val.csv', format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        )
        self.test_data = data.TabularDataset(
            path='./data/new_test.csv', format="csv", skip_header=True, 
            fields=[('ID', self.ID), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', None)]
        )
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.val_len = len(self.val_data)

        self.TEXT.build_vocab(self.train_data, self.test_data, vectors=pretrained_file, unk_init=torch.Tensor.normal_)
        self.LABEL.build_vocab(self.train_data)
        self.CATEGORIES_LABEL.build_vocab(self.train_data)
        self.unk_idx = self.TEXT.vocab.stoi[self.TEXT.unk_token]
        self.pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.trainloader = data.BucketIterator(self.train_data, sort_key=lambda x: len(x.Headline), batch_size=self.config['batch_size'], device=self.device, train=True, shuffle=True)
        self.valloader = data.BucketIterator(self.val_data, sort_key=lambda x: len(x.Headline), batch_size=self.config['batch_size'], device=self.device, train=False)

    def preprocess(self, train_file, val_file, test_file):
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)

        # process train categories
        replace_train = {
            'concussion': 'news',
            'travelnews': 'travel',
            'racing': 'formulaone',
            'gardening': 'home'
        }
        df_train = df_train.replace({'Category': replace_train})
        # df_train['Headline'] = df_train['Category'] + ' ' + df_train['Headline']
        df_val = df_val.replace({'Category': replace_train})
        # df_val['Headline'] = df_val['Category'] + ' ' + df_val['Headline']

        # process test categories
        replace_test = {
            'living': 'home',
            'middleeast': 'news',
            'us': 'news',
            'racing': 'formulaone'
        }
        df_test = df_test.replace({'Category': replace_test})
        # df_test['Headline'] = df_test['Category'] + ' ' + df_test['Headline']

        df_train.to_csv('./example/new_train.csv', index=False)
        df_val.to_csv('./example/new_val.csv', index=False)
        df_test.to_csv('./data/new_test.csv', index=False)