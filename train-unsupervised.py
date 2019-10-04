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
    train_loss = 0
    for batch_idx, (inputs, _) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), inputs.cuda()
        inputs, targets = Variable(inputs), nn.Upsample(size=128, mode='bilinear')(Variable(targets))

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

        train_loss += loss.item()

        print(batch_idx, len(trainloader), 'Loss: %.3f'
            % (train_loss/(batch_idx+1)))
    return train_loss/(batch_idx+1)

def test():
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, _) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), inputs.cuda()
        inputs, targets = Variable(inputs), nn.Upsample(size=128, mode='bilinear')(Variable(targets))

        # Forward
        if (args.variational):
            _, outputs, mu, logvar = net(inputs)
            loss = Loss(outputs, targets, mu, logvar)
        else:
            _, outputs = net(inputs)
            loss = Loss(outputs, targets)

        test_loss += loss.item()

        print(batch_idx, len(testloader), 'Loss: %.3f'
            % (test_loss/(batch_idx+1)))
    if epoch == args.epoch:
        print('Saving')
        if not os.path.isdir('{}'.format(args.path)):
            os.mkdir('{}'.format(args.path))
        torch.save(net.state_dict(), './{}/{}'.format(args.path, test_loss/(batch_idx+1)))

    if not os.path.exists(args.dc_path):
        os.mkdir(args.dc_path)
    save_image(targets.cpu().data, os.path.join(args.dc_path, 'inputs_{}.png'.format(epoch)), normalize=True, range=(-1, 1))
    save_image(outputs.cpu().data, os.path.join(args.dc_path, 'outputs_{}.png'.format(epoch)), normalize=True, range=(-1, 1))

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Deep Hashing')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    parser.add_argument('--epoch', type=int, default=128, metavar='epoch',
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

    torch.manual_seed(2)

    best_acc = 0
    start_epoch = 1
    transform_train = transforms.Compose(
        [transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[64], gamma=0.1)

    net.init_weights()
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
