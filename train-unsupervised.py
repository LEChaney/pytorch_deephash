import os
import shutil
import argparse

import torch
import torch.nn as nn

from net import VAEPlusLatent, AutoencoderPlusLatent, VAELoss

from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.optim.lr_scheduler

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, _) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), inputs.cuda()
            # noise = torch.cuda.FloatTensor(inputs.shape).normal_() * torch.cuda.FloatTensor(inputs.shape).uniform_(to=0.5)
        # else:
        #     noise = torch.randn(inputs.shape) * torch.rand(inputs.shape) * 0.5
        inputs, targets = Variable(inputs), nn.Upsample(size=128, mode='bilinear')(Variable(targets))
        # inputs += noise

        # Forward
        if (args.variational):
            _, outputs, mu, logvar = net(inputs)
            loss = Loss(outputs, targets, mu, logvar)
        else:
            _, outputs = net(inputs)
            loss = Loss(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(batch_idx, len(trainloader), 'Loss: %.3f' % loss.item())

def inv_normalize(y):
    mean = torch.tensor((0.4914, 0.4822, 0.4465))
    std = torch.tensor((0.2023, 0.1994, 0.2010))
    x = y.new(*y.size())
    x[:, 0, :, :] = y[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = y[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = y[:, 2, :, :] * std[2] + mean[2]
    return x

def test():
    net.eval()
    for batch_idx, (inputs, _) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), inputs.cuda()
            # noise = torch.cuda.FloatTensor(inputs.shape).normal_() * torch.cuda.FloatTensor(inputs.shape).uniform_(to=0.5)
        # else:
        #     noise = torch.randn(inputs.shape) * torch.rand(inputs.shape) * 0.5
        inputs, targets = Variable(inputs), nn.Upsample(size=128, mode='bilinear')(Variable(targets))
        # inputs += noise

        # Forward
        if (args.variational):
            _, outputs, mu, logvar = net(inputs)
            loss = Loss(outputs, targets, mu, logvar)
        else:
            _, outputs = net(inputs)
            loss = Loss(outputs, targets)

        print(batch_idx, len(testloader), 'Loss: %.3f' % loss.item())
    if epoch == args.epoch:
        print('Saving')
        if not os.path.isdir('{}'.format(args.path)):
            os.mkdir('{}'.format(args.path))
        torch.save(net.state_dict(), './{}/{}'.format(args.path, loss.item()))

    if not os.path.exists(args.dc_path):
        os.mkdir(args.dc_path)
    save_image(inv_normalize(inputs.cpu()).data, os.path.join(args.dc_path, 'inputs_{}.png'.format(epoch)), normalize=False)
    save_image(inv_normalize(outputs.cpu()).data, os.path.join(args.dc_path, 'outputs_{}.png'.format(epoch)), normalize=False)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.manual_seed(2)

    parser = argparse.ArgumentParser(description='Deep Hashing')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--epoch', type=int, default=16, metavar='epoch',
                        help='epoch')
    parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                        help='loading pretrained model(default = None)')
    parser.add_argument('--bits', type=int, default=128, metavar='bts',
                        help='binary bits')
    parser.add_argument('--path', type=str, default='model_unsupervised', metavar='P',
                        help='path directory')
    parser.add_argument('--dc_path', type=str, default='dc_img', metavar='DP',
                        help='output path for decoded images')
    parser.add_argument('--variational', action='store_true',
                        help='Use a Variational Autoencoder instead of a regular one')
    args = parser.parse_args()

    best_acc = 0
    start_epoch = 1
    transform_train = transforms.Compose(
        [transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=True, num_workers=2)

    if args.variational:
        net = VAEPlusLatent(args.bits)
        Loss = VAELoss()
    else:
        net = AutoencoderPlusLatent(args.bits)
        Loss = nn.MSELoss()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net.cuda()
        Loss = Loss.cuda()
        torch.backends.cudnn.deterministic = True

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)

    # net.init_weights()
    if args.pretrained:
        net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
        test()
    else:
        if os.path.isdir('{}'.format(args.path)):
            shutil.rmtree('{}'.format(args.path))
        for epoch in range(start_epoch, start_epoch+args.epoch):
            train(epoch)
            test()
            scheduler.step()
