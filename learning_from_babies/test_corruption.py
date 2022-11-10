# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
import sys

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
from torch.utils.data import TensorDataset, DataLoader
import resnet as RN
import pyramidnet as PYRM
from allcnn import AllConvNet
import resnet_cifar10 as RN_cifar
import numpy as np
import csv

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
parser.add_argument('-b', '--batch_size', default=256, type=int,
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
parser.add_argument('--no-verbose', dest='verbose', action='store_true',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--results', default='set/your/results/path', type=str)

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

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
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 10

            # Load corrupted data
            corrupted_data_dir = '/home/ankita/scratch/Datasets/CIFAR-10-C'
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
        if args.dataset == 'cifar10':
            model = RN_cifar.__dict__[args.net_type + f'{args.depth}']()
        else:
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

    # file to store the results
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    csv_filepath = f'{args.results}/corruptions.csv'

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    # err1, err5, val_loss = validate(val_loader, model, criterion)

    # evaluate on corrupted data
    if args.dataset.startswith('cifar'):
        validate_corrupted(model, criterion, corrupted_data_dir, csv_filepath, val_loader, normalize)
    elif args.dataset == 'tinyimagenet':
        validate_tin_corrupted(model, criterion, corrupted_data_dir, csv_filepath, val_loader, data_transforms['val'])

    # print('Accuracy (top-1 and 5 error):', err1, err5)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(input.min(), input.max())
    return top1.avg, top5.avg, losses.avg


def validate_corrupted(model, criterion, corrupted_data_dir, out_file, val_loader, normalize):

    distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]


    # set up filewriter
    f = (open(out_file, 'w'))
    writer = csv.writer(f)
    writer.writerow(['Corruption', 'mCE', 'CE-1', 'CE-2', 'CE-3', 'CE-4', 'CE-5'])

    # Evaluate on clean data
    err1, err5, val_loss = validate(val_loader, model, criterion)
    writer.writerow(['Clean data', err1])
    print('Clean data, top-1 error = ', err1)

    # Load labels
    labels = np.load(os.path.join(corrupted_data_dir, 'labels.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
    print('labels: ', labels.shape)

    # Evaluate on corrupted data
    error_rates = []
    for distortion_name in distortions:
        # rate_arr = show_performance(model, distortion_name, corrupted_data_dir)

        corr_data = np.load(os.path.join(corrupted_data_dir, distortion_name+'.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
        corr_data = np.swapaxes(corr_data, 1, 3)
        corr_data = np.swapaxes(corr_data, 2, 3)
        corr_data = corr_data / 255.
        # print('corr data', corr_data.shape)

        # print(distortion_name)
        errs = []
        for severity in range(5):
            x_corr = corr_data[10000*(severity):10000*(severity+1), :].astype('float')
            tensor_x = torch.Tensor(x_corr)
            tensor_x_transformed = normalize(tensor_x)
            
            tensor_y = torch.Tensor(labels[10000*(severity):10000*(severity+1)]).long()
            my_dataset = TensorDataset(tensor_x_transformed, tensor_y) # create your datset
            my_dataloader = DataLoader(my_dataset, num_workers=args.workers, 
                            batch_size = args.batch_size, pin_memory=True) # create your dataloader
            err1, err5, _ = validate(my_dataloader, model, criterion)
            # print(correct, error_top1)
            errs.append(err1)
            print(distortion_name, severity, err1)
            # break

        rate_mean = np.mean(errs)
        error_rates.append(rate_mean)
        writer.writerow([distortion_name, rate_mean, errs[0], errs[1], errs[2], errs[3], errs[4]])
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, rate_mean))

    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(np.mean(error_rates)))
    writer.writerow(['mCE', np.mean(error_rates)])

    f.close()


def validate_tin_corrupted(model, criterion, corrupted_data_dir, out_file, val_loader, val_transforms):

    distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression' ]

    # set up filewriter
    f = (open(out_file, 'w'))
    writer = csv.writer(f)
    writer.writerow(['Corruption', 'mCE', 'CE-1', 'CE-2', 'CE-3', 'CE-4', 'CE-5'])

    # Evaluate on clean data
    err1, err5, val_loss = validate(val_loader, model, criterion)
    writer.writerow(['Clean data', err1])
    print('Clean data, top-1 error = ', err1)

    # Evaluate on corrupted data
    error_rates = []
    for distortion_name in distortions:

        errs = []
        for severity in range(5):

            corr_dataset = datasets.ImageFolder(os.path.join(corrupted_data_dir, 
                            distortion_name, f'{severity+1}'), val_transforms) 
            val_loader = torch.utils.data.DataLoader(corr_dataset, batch_size=args.batch_size, 
                        shuffle=True,  num_workers=args.workers, pin_memory=True)

            err1, err5, _ = validate(val_loader, model, criterion)
            # print(correct, error_top1)
            errs.append(err1)
            print(distortion_name, severity, err1)
            # break

        rate_mean = np.mean(errs)
        error_rates.append(rate_mean)
        writer.writerow([distortion_name, rate_mean, errs[0], errs[1], errs[2], errs[3], errs[4]])
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, rate_mean))

    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(np.mean(error_rates)))
    writer.writerow(['mCE', np.mean(error_rates)])

    f.close()


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
    # print('pred = ', pred)
    # print('target = ', target)

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
