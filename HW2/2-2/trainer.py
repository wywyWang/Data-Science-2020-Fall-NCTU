import torch.nn as nn
import torch
import tqdm
import math
from transformermodel import TransformerModel
from attractivenet import AttractiveNet

class AttractiveTrainer:

    def __init__(self, config, device, train_loader, pretrained_embeddings):
        self.config = config
        
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.model = AttractiveNet(self.config).to(self.device)
        # self.model = TransformerModel(self.config).to(self.device)
        self.model.embedding.token.weight = nn.Parameter(pretrained_embeddings.to(self.device), requires_grad=False)

        # total parameters
        self.config['total_params'] = sum(p.numel() for p in self.model.parameters())
        self.config['total_learned_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': config['lr']['encoder']}, 
        #                                     {'params': self.model.embedding.parameters(), 'lr': config['lr']['embedding']},
        #                                     {'params': self.model.linear.parameters(), 'lr': config['lr']['linear'], 'weight_decay': 0.1},
        #                                  {'params': self.model.category_embedding.parameters(), 'lr': config['lr']['linear'], 'weight_decay': 0.1}])

        self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': config['lr']['encoder']}, 
                                            {'params': self.model.embedding.parameters(), 'lr': config['lr']['embedding']},
                                         {'params': self.model.category_embedding.parameters(), 'lr': config['lr']['linear']}], lr=config['lr']['linear'])

        self.train_loader = train_loader

    def train(self):
        for epoch in tqdm.tqdm(range(self.config['epochs']), desc='Epoch: '):
            # print("Epoch {}".format(epoch))
            self.train_predict, self.train_true = self.iteration(epoch)
            # self.scheduler.step()
        self.save(self.config['save_name'], self.config['timestr'], self.config['epochs'], self.train_loss)

    def iteration(self, epoch):
        self.model.train()
        # data_iter = tqdm.tqdm(enumerate(self.train_loader),
        #                     desc="EP:{} | lr: {}".format(epoch, self.lr),
        #                     total=len(self.train_loader),
        #                     bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0
        for i, data in enumerate(self.train_loader):
            inputs = data.Headline
            attractive_labels = data.Label
            attractive_categories = data.Category

            # forward masked_lm model
            attractive_prediction = self.model(inputs, attractive_categories)

            # print(inputs, flush=True)
            # print(attractive_labels, flush=True)
            # print(attractive_prediction, flush=True)
            # # print(attractive_categories, flush=True)
            # print(inputs.shape, flush=True)
            # print(attractive_labels.shape, flush=True)
            # print(attractive_prediction.shape, flush=True)
            # print(attractive_categories.shape, flush=True)
            # print(self.criterion(attractive_prediction, attractive_labels).item(), flush=True)
            # 1/0

            # NLLLoss of predicting masked token
            loss = self.criterion(attractive_prediction, attractive_labels)

            # backward and optimize in training stage
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1)
            }

            if i % self.config['log_steps'] == 0:
                with open('log/{}_{}_train'.format(self.config['timestr'], epoch), 'a') as f_train:
                    f_train.write(str(post_fix) + '\n')

        # evaluate training accuracy
        attractive_predict, attractive_true, self.train_loss = self.evaluate(self.train_loader, 'train')
        return attractive_predict, attractive_true

    def evaluate(self, data_loader, str_code):
        self.model.eval()
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                     desc="EP: {} | lr: {}".format(str_code, self.config['lr']),
        #                     total=len(data_loader),
        #                     bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0

        attractive_predict = torch.Tensor().to(self.device)
        attractive_true = torch.Tensor().to(self.device)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs = data.Headline
                attractive_labels = data.Label
                attractive_categories = data.Category

                # forward masked_lm model
                attractive_prediction = self.model(inputs, attractive_categories)

                # MSELoss
                loss = self.criterion(attractive_prediction, attractive_labels)

                avg_loss += loss.item()

                # _, predict_class = torch.max(attractive_prediction, dim=1)
                # # print(predict_class)

                attractive_predict = torch.cat((attractive_predict, attractive_prediction))
                attractive_true = torch.cat((attractive_true, attractive_labels))


        avg_loss /= len(data_loader)
        print()
        print("EP_{} | avg_loss: {} |".format(str_code, avg_loss))

        return attractive_predict.cpu().detach().tolist(), attractive_true.cpu().detach().tolist(), avg_loss

    def save(self, prefix_name, timestr, epochs, loss):
        output_name = './model/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)
        torch.save(self.model.state_dict(), output_name)

        # store config parameters
        config_name = './config/' + prefix_name + '_' + str(timestr) + '_' + str('{:.4f}'.format(loss)) + '.' + str(epochs)

        with open(config_name, 'w') as config_file:
            config_file.write(str(self.config))