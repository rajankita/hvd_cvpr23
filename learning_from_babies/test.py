# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
from allcnn import AllConvNet

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Test')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, pyamidnet and allconv')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--sigmas', nargs='+', type=float,
                    help = 'sigma for evaluation on blurred images')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc1 = 100
best_acc5 = 100


def main():
    global args, best_acc1, best_acc5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':

        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    elif args.dataset == 'imagenet-sketch':

        valdir = os.path.join('/home/ankita/scratch/Datasets/imagenet-sketch')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    elif args.dataset == 'tinyimagenet':
        data_dir = '../datasets/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomRotation(20),
                transforms.RandomAffine(18, translate=[0.2, 0.2], shear=0.15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
        }
        # train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']) 
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val_labeled'), data_transforms['val']) 

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                        # shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=True,  num_workers=args.workers, pin_memory=True)

        numberofclass = 200

        # Load corrupted data
        corrupted_data_dir = '../datasets/Tiny-ImageNet-C'

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    elif args.net_type == 'allconv':
        model = AllConvNet(numberofclass)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(args.pretrained))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    for sigma in args.sigmas:
        acc1, acc5, val_loss = validate(val_loader, model, criterion, sigma)
        print(f'Sigma: {sigma}, acc = {acc1}, err = {100-acc1}')

    print('Accuracy (top-1 and 5 acc):', acc1, acc5)
    print('Top-1 error: ', 100. - acc1)


class Blur(object):
    """Blur the image in a sample.
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self.kernel = int(self.sigma*6 + 1)

    def __call__(self, sample):
        # apply blur
        if self.sigma > 0:
            sample = transforms.functional.gaussian_blur(sample, self.kernel, self.sigma) 

        return sample


def validate(val_loader, model, criterion, sigma):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # define blur transform
    blur = Blur(sigma)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        input = input.cuda()
        input = blur(input)

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        top1_err = 100. - top1.avg

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        # wrong_k = batch_size - correct_k
        # res_acc.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
