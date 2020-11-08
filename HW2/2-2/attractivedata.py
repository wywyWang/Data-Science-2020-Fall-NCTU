import pandas as pd
import numpy as np
import torch
import torchtext
from torchtext import data
from sklearn.model_selection import KFold


class AttractiveData:
    """
    Read data and use TorchText to process to BucketIterator
    Args:
        train_file: training filename
        val_file: validation filename
        test_file: testing filename
        pretrained_file: pretrained static word embedding name
        config: config setting
    """
    def __init__(self, train_file, val_file, test_file, pretrained_file, config):
        self.config = config
        self.preprocess(train_file, val_file, test_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df_train = pd.read_csv('./data/new_train.csv')
        # self.df_val = pd.read_csv('./example/new_val.csv')
        self.df_test = pd.read_csv('./data/new_test.csv')

        self.TEXT = data.Field(sequential=True, lower=True, batch_first=True)
        self.CATEGORIES_LABEL = data.LabelField(sequential=False, batch_first=True)
        self.LABEL = data.Field(dtype=torch.float, sequential=False, use_vocab=False, batch_first=True)
        self.ID = data.Field(sequential=False, use_vocab=False, batch_first=True)

        self.train_field = [('ID', None), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', self.LABEL)]
        self.test_field = [('ID', self.ID), ('Headline', self.TEXT), ('Category', self.CATEGORIES_LABEL), ('Label', None)]

        self.train_data = data.TabularDataset(
            path='./data/new_train.csv', format="csv", skip_header=True, 
            fields=self.train_field
        )
        self.val_data = data.TabularDataset(
            path='./data/new_val.csv', format="csv", skip_header=True, 
            fields=self.train_field
        )
        self.test_data = data.TabularDataset(
            path='./data/new_test.csv', format="csv", skip_header=True, 
            fields=self.test_field
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
        df_val = df_val.replace({'Category': replace_train})

        # process test categories
        replace_test = {
            'living': 'home',
            'middleeast': 'news',
            'us': 'news',
            'racing': 'formulaone'
        }
        df_test = df_test.replace({'Category': replace_test})

        df_train.to_csv('./data/new_train.csv', index=False)
        df_val.to_csv('./data/new_val.csv', index=False)
        df_test.to_csv('./data/new_test.csv', index=False)

    def k_fold_data(self):
        """
        k fold training set, but unused
        """
        kf = KFold(n_splits=self.config['n_splits'], shuffle=True, random_state=1234)
        train_data_arr = np.array(self.train_data.examples)
        for train_index, val_index in kf.split(train_data_arr):
            yield(
                data.Dataset(train_data_arr[train_index], fields=self.train_field), 
                data.Dataset(train_data_arr[val_index], fields=self.train_field)
            )