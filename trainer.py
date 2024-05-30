import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.get_device()
        self.train_loader, self.test_loader = self.get_data_loaders()
        self.model = Net().to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)
        self.best_loss = float('inf')

    def get_device(self):
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        use_mps = not self.args.no_mps and torch.backends.mps.is_available()
        if use_cuda:
            return torch.device("cuda")
        elif use_mps:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def get_data_loaders(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_kwargs = {'batch_size': self.args.batch_size}
        test_kwargs = {'batch_size': self.args.test_batch_size}
        if self.device == torch.device("cuda"):
            cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        return train_loader, test_loader

    def train(self, epoch):
        train_loss = 0
        correct = 0
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}"):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(self.train_loader)
        print("Training Loss:", train_loss)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)}'
              f' ({100. * correct / len(self.test_loader.dataset):.0f}%)')
        
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            torch.save(self.model.state_dict(), "model.pt")
            print("Model saved")

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)
            self.test()
            self.scheduler.step()

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()