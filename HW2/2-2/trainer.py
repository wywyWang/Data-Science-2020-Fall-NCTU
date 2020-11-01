import torch.nn as nn
import torch
import tqdm
import math
from attractivenet import AttractiveNet

class AttractiveTrainer:

    def __init__(self, config, device, train_loader, pretrained_embeddings):
        self.config = config
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.device = device
        self.model = AttractiveNet(self.config).to(self.device)
        self.model.embedding.token.weight = nn.Parameter(pretrained_embeddings.to(self.device), requires_grad=False)
        self.model.embedding.token.weight.data[0] = torch.zeros(300)
        self.model.embedding.token.weight.data[1] = torch.zeros(300)

        # total parameters
        self.config['total_params'] = sum(p.numel() for p in self.model.parameters())
        self.config['total_learned_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # self.optimizer = torch.optim.SGD([
        #     {'params': self.model.encoder.parameters(), 'lr': config['lr']['encoder']},
        #     {'params': self.model.linear.parameters()},
        #     # {'params': self.model.cnn1.parameters()},
        #     # {'params': self.model.cnn2.parameters()},
        # ], lr=config['lr']['linear'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr']['linear'])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr']['linear'], momentum=0.9)
        # self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': config['lr']['encoder']}], lr=config['lr']['linear'])
        # self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': config['lr']['encoder']}, 
        #                                     {'params': self.model.embedding.parameters(), 'lr': config['lr']['embedding']},
        #                                  {'params': self.model.category_embedding.parameters(), 'lr': config['lr']['linear']}], lr=config['lr']['linear'])

        self.train_loader = train_loader

    def train(self):
        for epoch in tqdm.tqdm(range(self.config['epochs']), desc='Epoch: '):
            self.iteration(epoch)
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
        self.train_loss = self.evaluate(self.train_loader, 'train')

    def evaluate(self, data_loader, str_code):
        self.model.eval()
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                     desc="EP: {} | lr: {}".format(str_code, self.config['lr']),
        #                     total=len(data_loader),
        #                     bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0

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

                avg_loss += loss.item()

        avg_loss /= self.config['train_len']
        # avg_loss /= len(data_loader)
        print()
        print("EP_{} | avg_loss: {} |".format(str_code, avg_loss))

        return avg_loss

    def save(self, prefix_name, timestr, epochs, loss):
        output_name = './model/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)
        torch.save(self.model.state_dict(), output_name)

        # store config parameters
        config_name = './config/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)

        with open(config_name, 'w') as config_file:
            config_file.write(str(self.config))