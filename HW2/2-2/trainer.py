import torch.nn as nn
import torch
import tqdm
import math
from attractivenet import AttractiveNet

class AttractiveTrainer:

    def __init__(self, config, device, train_loader, val_loader, pretrained_embeddings):
        self.config = config
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.device = device
        self.model = AttractiveNet(self.config).to(self.device)
        self.model.embedding.token.weight = nn.Parameter(pretrained_embeddings.to(self.device), requires_grad=True)
        # self.model.embedding.token.weight.data[0] = torch.zeros(300)
        # self.model.embedding.token.weight.data[1] = torch.zeros(300)

        # total parameters
        self.config['total_params'] = sum(p.numel() for p in self.model.parameters())
        self.config['total_learned_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.optimizer = torch.optim.SGD([
            # {'params': self.model.encoder_origin.parameters()}, 
            {'params': self.model.encoder_bigram.parameters()}, 
            {'params': self.model.encoder_trigram.parameters()}, 
            {'params': self.model.bigramcnn.parameters()}, 
            {'params': self.model.trigramcnn.parameters()},
            {'params': self.model.linear.parameters()}, 
            {'params': self.model.embedding.parameters(), 'lr': config['lr']['embedding']},
        ], lr=config['lr']['linear'], momentum=0.9)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        for epoch in tqdm.tqdm(range(self.config['epochs']), desc='Epoch: '):
            self.iteration(epoch)
            if epoch % 10 == 0 and epoch != 0:
                self.save(self.config['save_name'], self.config['timestr'], epoch, self.train_loss)
        self.save(self.config['save_name'], self.config['timestr'], self.config['epochs'], self.train_loss)

    def iteration(self, epoch):
        self.model.train()
        
        avg_loss = 0.0
        print("====")
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

            # print(attractive_prediction.view(-1))
            # print(attractive_labels)
            # print(self.criterion(attractive_prediction.view(-1), attractive_labels))
            # 1/0

            # loss
            loss = self.criterion(attractive_prediction.view(-1), attractive_labels)

            # backward and optimize in training stage
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1) / self.config['batch_size']
            }

            # if i % self.config['log_steps'] == 0:
                # with open('log/{}_{}_train'.format(self.config['timestr'], epoch), 'a') as f_train:
                #     f_train.write(str(post_fix) + '\n')

        # evaluate training accuracy
        self.train_loss = self.evaluate(self.train_loader, self.val_loader, 'train')

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
                
                # print(attractive_prediction.view(-1)[0:3], attractive_labels[0:3], loss)

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
                
                # print(attractive_prediction.view(-1)[0:3], attractive_labels[0:3], loss)

                val_loss += loss.item()

        val_loss /= self.config['val_len']

        print()
        print("EP_{} | train loss: {} | val loss: {} |".format(str_code, train_loss, val_loss))

        with open('log/{}'.format(self.config['timestr']), 'a') as f_train:
            f_train.write(str(train_loss) + ', ' + str(val_loss) + '\n')

        return train_loss

    def save(self, prefix_name, timestr, epochs, loss):
        output_name = './model/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)
        torch.save(self.model.state_dict(), output_name)

        if epochs == self.config['epochs']:
            # store config parameters
            config_name = './config/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)

            with open(config_name, 'w') as config_file:
                config_file.write(str(self.config))