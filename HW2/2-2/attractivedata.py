import pandas as pd
import torch
import torchtext
from torchtext import data


class AttractiveData:
    def __init__(self, train_file, test_file, pretrained_file, config):
        self.config = config
        self.preprocess(train_file, test_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df_train = pd.read_csv('./data/new_train.csv')
        self.df_test = pd.read_csv('./data/new_test.csv')

#         self.TEXT = data.Field(sequential=True, init_token='<s>', lower=False, tokenize=self.tokenizer, fix_length=max_size, pad_token='0')
        # self.TEXT = data.Field(sequential=True, lower=False, tokenize=self.tokenizer, fix_length=self.config['max_size'], pad_token='0')
        self.TEXT = data.Field(sequential=True, lower=True, fix_length=self.config['max_seq'], pad_token='0')
        # self.LABEL = data.LabelField(dtype=torch.long, sequential=False)
        self.CATEGORIES_LABEL = data.LabelField(sequential=False)
        self.LABEL = data.Field(dtype=torch.float, sequential=False, use_vocab=False)
        # self.CATEGORIES_LABEL = data.Field(sequential=False, use_vocab=False)

        self.train_data = data.TabularDataset(
            path='./data/new_train.csv', format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        )
        self.test_data = data.TabularDataset(
            path='./data/new_test.csv', format="csv", skip_header=True, 
            fields=[('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', None)]
        )

        self.TEXT.build_vocab(self.train_data, self.test_data, vectors=pretrained_file, min_freq=self.config['min_freq'])
        self.LABEL.build_vocab(self.train_data)
        self.CATEGORIES_LABEL.build_vocab(self.train_data)

        self.trainloader = data.BucketIterator(self.train_data, sort_key=lambda x: len(x.Text), batch_size=self.config['batch_size'], device=self.device, train=True, shuffle=True)

    def preprocess(self, train_file, test_file):
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        # eliminate train mean
        df_train.Label -= 3.15

        # process train categories
        replace_train = {
            'concussion': 'news',
            'travelnews': 'travel',
            'racing': 'formulaone',
            'gardening': 'home'
        }
        df_train = df_train.replace({'Category': replace_train})

        # process test categories
        replace_test = {
            'living': 'home',
            'middleeast': 'news',
            'us': 'news',
            'racing': 'formulaone'
        }
        df_test = df_test.replace({'Category': replace_test})

        df_train.to_csv('./data/new_train.csv', index=False)
        df_test.to_csv('./data/new_test.csv', index=False)