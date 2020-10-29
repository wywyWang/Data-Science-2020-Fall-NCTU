import torch.nn as nn
import torch
import tqdm
import math
from transformermodel import TransformerModel

class AttractiveTrainer:

    def __init__(self, save_name, log_steps, epochs, lr, timestr, device, train_loader, input_dim, category_dim, category_embedding_dim, embedding_dim, hidden_dim, output_dim, pretrained_embeddings, dropout, num_layers, nhead):
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_name = save_name
        self.log_steps = log_steps
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = TransformerModel(nhead, input_dim, category_dim, category_embedding_dim, embedding_dim, hidden_dim, output_dim, dropout, num_layers).to(self.device)
        self.model.embedding.token.weight = nn.Parameter(pretrained_embeddings.to(self.device), requires_grad=False)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.SGD([{'params': self.model.transformer_encoder.parameters(), 'lr': 1e-5}, 
                                            {'params': self.model.embedding.parameters(), 'lr': 1e-5},
                                            {'params': self.model.linear.parameters(), 'lr': 1e-3}])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # self.scheduler = torch.optim.ScheduledOptim(self.optimizer, self.model.hidden, n_warmup_steps=50)
        self.timestr = timestr
        self.train_loader = train_loader

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch))
            self.train_predict, self.train_true = self.iteration(epoch)
            # self.scheduler.step()
        self.save(self.timestr, self.epochs, self.train_loss)

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

            if i % self.log_steps == 0:
                with open('log/{}_{}_train'.format(self.timestr, epoch), 'a') as f_train:
                    f_train.write(str(post_fix) + '\n')

        # evaluate training accuracy
        attractive_predict, attractive_true, self.train_loss = self.evaluate(self.train_loader, 'train')
        return attractive_predict, attractive_true

    def evaluate(self, data_loader, str_code):
        self.model.eval()
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="EP: {} | lr: {}".format(str_code, self.lr),
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0

        attractive_predict = torch.Tensor().to(self.device)
        attractive_true = torch.Tensor().to(self.device)

        with torch.no_grad():
            for i, data in data_iter:
                inputs = data.Headline
                attractive_labels = data.Label
                attractive_categories = data.Category

                # forward masked_lm model
                attractive_prediction = self.model(inputs, attractive_categories)

                # MSELoss
                loss = self.criterion(attractive_prediction, attractive_labels)

                avg_loss += loss.item()

                _, predict_class = torch.max(attractive_prediction, dim=1)
                # print(predict_class)

                attractive_predict = torch.cat((attractive_predict, predict_class))
                attractive_true = torch.cat((attractive_true, attractive_labels))


        avg_loss /= len(data_iter)
        print()
        print("EP_{} | avg_loss: {} |".format(str_code, avg_loss))

        return attractive_predict.cpu().detach().tolist(), attractive_true.cpu().detach().tolist(), avg_loss

    def save(self, timestr, epochs, loss):
        output_name = self.save_name + '_' + str(timestr) + '_' + str('{:.3f}'.format(loss)) + '.' + str(epochs)
        torch.save(self.model.state_dict(), output_name)