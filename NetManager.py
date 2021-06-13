import os
import torch
from tqdm import tqdm
from NetArchitecture import NetArchitecture
from NetDataset import NetDataset

# file_train = open("Loss_Train3.txt","w")
# file_validation = open("Loss_Validation3.txt","w")
class NetManager:
    def __init__(self):
        self.model = NetArchitecture()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.print_every = 50

    def load(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def save(self, filename):
        state = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def set_device(self, device):
        if torch.cuda.is_available():
            self.device = torch.device(device)
        self.model.to(self.device)

    def train(self, train_input, train_groundtruth, valid_input, valid_groundtruth, max_epoch):
        train_dataset = NetDataset(train_input, train_groundtruth)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

        valid_dataset = NetDataset(valid_input, valid_groundtruth)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        train_best_loss = 1.0e99 # self.__validate(train_loader)
        valid_best_loss = self.__validate(valid_loader)
        for epoch in range(max_epoch):
            train_loss = self.__train_epoch(train_loader)
            if train_loss < train_best_loss:
                train_best_loss = train_loss
                self.save('bestModel_training.pt')
                # file_train.write(str(train_best_loss))
                # file_train.write(",")
                
                print('Epoch {}: New best training loss: {}'.format(epoch, train_best_loss))
            else:
                print('Epoch {}: Current training loss: {}. Best loss: {}'.format(epoch, train_loss, train_best_loss))

            valid_loss = self.__validate(valid_loader)
            if valid_loss < valid_best_loss:
                valid_best_loss = valid_loss
                self.save('bestModel_validation.pt')
                # file_validation.write(str(valid_best_loss))
                # file_validation.write(",")
                
                print('Epoch {}: New best validation loss: {}'.format(epoch, valid_best_loss))
            else:
                print('Epoch {}: Current validation loss: {}. Best loss: {}'.format(epoch, valid_loss, valid_best_loss))

    def __train_epoch(self, train_loader):
        self.model.train()
        torch.enable_grad()
        train_loss = 0
        count = len(train_loader)
        for i, (batch, label) in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()
            output = self.model(batch.to(self.device))

            loss = self.criterion(output, label.to(self.device))
            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.print_every == 0:
                tqdm.write('Sample {}/{}. Current training loss: {}'.format(i + 1, count, train_loss / i))
        train_loss /= count
        return train_loss

    def __validate(self, valid_loader):
        self.model.eval()
        torch.no_grad()
        eval_loss = 0
        count = len(valid_loader)
        for i, (batch, label) in enumerate(tqdm(valid_loader)):
            output = self.model(batch.to(self.device))

            loss = self.criterion(output, label.to(self.device))
            eval_loss += loss.item()

            if (i + 1) % self.print_every == 0:
                tqdm.write('Sample {}/{}. Current validation loss: {}'.format(i + 1, count, eval_loss / i))
        eval_loss /= count
        return eval_loss

    def test(self, test_input, test_output="results"):
        test_dataset = NetDataset(test_input)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        self.model.eval()
        torch.no_grad()
        count = len(test_loader)
        for i, (batch, _) in enumerate(tqdm(test_loader)):
            output = self.model(batch.to(self.device))
            test_dataset.save_output(output.to("cpu"), \
                    os.path.join(test_output, test_dataset.get_filename(i)))

            if (i + 1) % self.print_every == 0:
                tqdm.write('Sample {}/{}'.format(i + 1, count))

