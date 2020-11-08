import torch.nn as nn
import torch
import tqdm
import math
import os
from torchtext import data
from attractivenet import AttractiveNet

class AttractiveTrainer:

    def __init__(self, config, device, train_loader, val_loader, pretrained_embeddings):
        self.config = config
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.device = device
        self.model = AttractiveNet(self.config).to(self.device)
        self.model.embedding.token.weight = nn.Parameter(pretrained_embeddings.to(self.device), requires_grad=True)

        # total parameters
        self.config['total_params'] = sum(p.numel() for p in self.model.parameters())
        self.config['total_learned_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.optimizer = torch.optim.SGD([
            {'params': self.model.encoder_bigram.parameters()}, 
            {'params': self.model.encoder_trigram.parameters()}, 
            {'params': self.model.bigramcnn.parameters()}, 
            {'params': self.model.trigramcnn.parameters()},
            {'params': self.model.linear.parameters()}, 
            # {'params': self.model.category_embedding.parameters()}, 
            {'params': self.model.embedding.parameters(), 'lr': config['lr']['embedding']},
        ], lr=config['lr']['linear'], momentum=0.9)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.folder_name = './model/' + self.config['save_name'] + '_' + self.config['timestr'] + '/'
        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def train(self):
        for epoch in tqdm.tqdm(range(self.config['epochs']), desc='Epoch: '):
            self.train_loss, self.val_loss = self.iteration(epoch)
            if epoch > 30 and epoch < 200:
                if epoch % 5 == 0:
                    self.save(self.config['save_name'], self.config['timestr'], epoch, self.train_loss)
            else:
                if epoch % 10 == 0 and epoch != 0:
                    self.save(self.config['save_name'], self.config['timestr'], epoch, self.train_loss)
            print()
            print("EP_{} | train loss: {} | val loss: {} |".format(epoch, self.train_loss, self.val_loss))
        self.save(self.config['save_name'], self.config['timestr'], self.config['epochs'], self.train_loss)

    def iteration(self, epoch):
        self.model.train()
        
        avg_loss = 0.0
        for i, data in enumerate(self.train_loader):
            inputs = data.Headline
            attractive_labels = data.Label
            attractive_categories = data.Category

            # forward
            # with open('debug', 'a') as f_train:
            #     # f_train.write(str(inputs) + '\n')
            #     # f_train.write(str(attractive_labels) + '\n')
            #     for parameters in self.model.linear[2].parameters():
            #         f_train.write(str(parameters) + '\n')
            #     f_train.write(str('==============') + '\n')
            attractive_prediction = self.model(inputs, attractive_categories, phase='train')

            # loss
            loss = self.criterion(attractive_prediction.view(-1), attractive_labels)

            # backward and optimize in training stage
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

            # post_fix = {
            #     "epoch": epoch,
            #     "iter": i,
            #     "avg_loss": avg_loss / (i + 1) / self.config['batch_size']
            # }

            # if i % self.config['log_steps'] == 0:
                # with open('log/{}_{}_train'.format(self.config['timestr'], epoch), 'a') as f_train:
                #     f_train.write(str(post_fix) + '\n')

        # evaluate training accuracy
        train_loss, val_loss = self.evaluate(self.train_loader, self.val_loader, 'train')
        return train_loss, val_loss

    def evaluate(self, data_loader, val_data_loader, str_code):
        self.model.eval()
    
        train_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs = data.Headline
                attractive_labels = data.Label
                attractive_categories = data.Category

                # forward
                attractive_prediction = self.model(inputs, attractive_categories, phase='train')

                # MSELoss
                loss = self.criterion(attractive_prediction.view(-1), attractive_labels)

                train_loss += loss.item()

        train_loss /= self.config['train_len']

        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_data_loader):
                inputs = data.Headline
                attractive_labels = data.Label
                attractive_categories = data.Category

                # forward
                attractive_prediction = self.model(inputs, attractive_categories, phase='train')

                # MSELoss
                loss = self.criterion(attractive_prediction.view(-1), attractive_labels)

                val_loss += loss.item()

        val_loss /= self.config['val_len']

        # print()
        # print("EP_{} | train loss: {} | val loss: {} |".format(str_code, train_loss, val_loss))

        with open('log/{}'.format(self.config['timestr']), 'a') as f_train:
            f_train.write(str(train_loss) + ', ' + str(val_loss) + '\n')

        return train_loss, val_loss

    def save(self, prefix_name, timestr, epochs, loss):
        output_name = self.folder_name + str('{:.6f}'.format(loss)) + '.' + str(epochs)
        torch.save(self.model.state_dict(), output_name)

        # store config parameters
        config_name = './config/' + prefix_name + '_' + str(timestr) + '_' + str('{:.6f}'.format(loss)) + '.' + str(epochs)

        with open(config_name, 'w') as config_file:
            config_file.write(str(self.config))