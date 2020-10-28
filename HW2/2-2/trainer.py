import torch.nn as nn
import torch
import tqdm
import math
# from attractivenet import AttractiveNet
from transformermodel import TransformerModel

class AttractiveTrainer:

    def __init__(self, save_name, log_steps, epochs, lr, timestr, device, train_loader, test_loader, input_dim, embedding_dim, hidden_dim, output_dim, pretrained_embeddings, dropout, num_layers, nhead):
        self.criterion = torch.nn.MSELoss()
        self.save_name = save_name
        self.log_steps = log_steps
        self.epochs = epochs
        self.lr = lr
        self.device = device
        # self.model = AttractiveNet(input_dim, embedding_dim, hidden_dim, output_dim, dropout, num_layers).to(self.device)
        self.model = TransformerModel(nhead, input_dim, embedding_dim, hidden_dim, output_dim, dropout, num_layers).to(self.device)
        self.model.embedding.token.weight.data = pretrained_embeddings.cuda()
        self.model.embedding.token.weight.requires_grad = False                 # freeze embedding

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # self.scheduler = torch.optim.ScheduledOptim(self.optimizer, self.model.hidden, n_warmup_steps=50)
        self.timestr = timestr
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        for epoch in range(self.epochs):
            self.iteration(epoch)
            # self.scheduler.step()
            self.save(epoch)

    def iteration(self, epoch):
        self.model.train()
        data_iter = tqdm.tqdm(enumerate(self.train_loader),
                            desc="EP:{} | lr: {}".format(epoch, self.lr),
                            total=len(self.train_loader),
                            bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0
        for i, data in data_iter:
            inputs = data.Headline
            attractive_labels = data.Label

            # forward masked_lm model
            attractive_prediction = self.model(inputs)

            # print(inputs)
            # print(attractive_labels)
            # print(attractive_prediction)
            # print(inputs.shape)
            # print(attractive_labels.shape)
            # print(attractive_prediction.shape)
            # print(self.criterion(attractive_prediction, attractive_labels).item())
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
        self.evaluate(self.train_loader, 'train')

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

                # forward masked_lm model
                attractive_prediction = self.model(inputs)

                # MSELoss
                loss = self.criterion(attractive_prediction, attractive_labels)

                avg_loss += loss.item()

                attractive_predict = torch.cat((attractive_predict, attractive_prediction[0]))
                attractive_true = torch.cat((attractive_true, attractive_labels))


        avg_loss /= len(data_iter)
        print()
        print("EP_{} | avg_loss: {} |".format(str_code, avg_loss))

        return attractive_predict.cpu().detach().tolist(), attractive_true.cpu().detach().tolist()

    def save(self, epoch):
        output_name = self.save_name + '.' + str(epoch)
        torch.save(self.model.state_dict(), output_name)