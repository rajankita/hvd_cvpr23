# # original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

# import argparse
# import os
# import shutil
# import time
# import sys

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# import resnet as RN
# import pyramidnet as PYRM
# from allcnn import AllConvNet
# import resnet_cifar10 as RN_cifar
# import utils
# import numpy as np
# from PIL import Image
# import random
# from copy import deepcopy

# import warnings

# warnings.filterwarnings("ignore")

# seed = 0    
# os.environ['PYTHONHASHSEED'] = str(seed)
# # Torch RNG
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# # Python RNG
# np.random.seed(seed)
# random.seed(seed)
# # torch.set_deterministic(True)
# # torch.use_deterministic_algorithms(True)

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

# parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
# parser.add_argument('--net_type', default='pyramidnet', type=str,
#                     help='networktype: resnet, and pyamidnet')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# # parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     # help='number of total epochs to run')
# parser.add_argument('-b', '--batch_size', default=128, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=100, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--depth', default=32, type=int,
#                     help='depth of the network (default: 32)')
# parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
#                     help='to use basicblock for CIFAR datasets (default: bottleneck)')
# parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
#                     help='dataset (options: cifar10, cifar100, and imagenet)')
# parser.add_argument('--no-verbose', dest='verbose', action='store_false',
#                     help='to print the status at every iteration')
# parser.add_argument('--alpha', default=300, type=float,
#                     help='number of new channel increases per depth (default: 300)')
# parser.add_argument('--expname', default='TEST', type=str,
#                     help='name of experiment')
# parser.add_argument('--beta', default=0, type=float,
#                     help='hyperparameter beta')
# parser.add_argument('--cutmix_prob', default=0, type=float,
#                     help='cutmix probability')
# # curriculum related
# parser.add_argument('--epochs', nargs='+', type=int, 
#                     help = 'epoch segments in the curriculum')
# parser.add_argument('--sigmas', nargs='+', type=float,
#                     help = 'sigma for curriculum segments')

# parser.set_defaults(bottleneck=True)
# parser.set_defaults(verbose=True)

# best_acc1 = 0
# best_acc5 = 0


# def main():

#     # torch.manual_seed(1)
#     # np.random.seed(1)

#     global args, best_acc1, best_acc5
#     args = parser.parse_args()
    
#     args.net_type = 'allconv'
#     args.dataset = 'cifar10'
#     args.batch_size = 128
#     args.lr = 0.1
#     args.expname = 'runs/runs_2022/cifar10/allconv/clewr_freeze'
#     args.epochs = [1, 1, 96]
#     args.sigmas = [2, 1, 0]

#     if args.dataset.startswith('cifar'):
#         normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])

#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             normalize
#         ])

#         if args.dataset == 'cifar100':
#             train_loader = torch.utils.data.DataLoader(
#                 datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
#             val_loader = torch.utils.data.DataLoader(
#                 datasets.CIFAR100('../data', train=False, transform=transform_test),
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
#             numberofclass = 100
#         elif args.dataset == 'cifar10':
#             train_loader = torch.utils.data.DataLoader(
#                 datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
#             val_loader = torch.utils.data.DataLoader(
#                 datasets.CIFAR10('../data', train=False, transform=transform_test),
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
#             numberofclass = 10
#         else:
#             raise Exception('unknown dataset: {}'.format(args.dataset))

#     elif args.dataset == 'imagenet':
#         traindir = os.path.join('/home/data/ILSVRC/train')
#         valdir = os.path.join('/home/data/ILSVRC/val')
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

#         jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
#                                       saturation=0.4)
#         lighting = utils.Lighting(alphastd=0.1,
#                                   eigval=[0.2175, 0.0188, 0.0045],
#                                   eigvec=[[-0.5675, 0.7192, 0.4009],
#                                           [-0.5808, -0.0045, -0.8140],
#                                           [-0.5836, -0.6948, 0.4203]])

#         train_dataset = datasets.ImageFolder(
#             traindir,
#             transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 jittering,
#                 lighting,
#                 normalize,
#             ]))

#         train_sampler = None

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#             num_workers=args.workers, pin_memory=True, sampler=train_sampler)

#         val_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(valdir, transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize,
#             ])),
#             batch_size=args.batch_size, shuffle=False,
#             num_workers=args.workers, pin_memory=True)
#         numberofclass = 1000

#     elif args.dataset == 'tinyimagenet':
#         data_dir = '../datasets/tiny-imagenet-200/'
#         data_transforms = {
#             'train': transforms.Compose([
#                 # transforms.RandomRotation(20),
#                 transforms.RandomAffine(18, translate=[0.2, 0.2], shear=0.15),
#                 transforms.RandomHorizontalFlip(0.5),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
#             ]),
#             'val': transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
#             ]),
#         }
#         train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']) 
#         val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val_labeled'), data_transforms['val']) 

#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
#                         shuffle=True, num_workers=args.workers, pin_memory=True)
#         val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
#                         shuffle=True,  num_workers=args.workers, pin_memory=True)

#         numberofclass = 200

#     else:
#         raise Exception('unknown dataset: {}'.format(args.dataset))

#     print("=> creating model '{}'".format(args.net_type))
#     if args.net_type == 'resnet':
#         if args.dataset == 'cifar10':
#             model = RN_cifar.__dict__[args.net_type + f'{args.depth}']()
#         else:
#             model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
#     elif args.net_type == 'pyramidnet':
#         model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
#                                 args.bottleneck)
#     elif args.net_type == 'allconv':
#         model = AllConvNet(numberofclass)
#     else:
#         raise Exception('unknown network architecture: {}'.format(args.net_type))

#     model = torch.nn.DataParallel(model).cuda()

#     print(model)
#     print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()

#     optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay, nesterov=True)

#     # define learning rate schedule
#     # decayRate = 0.97
#     # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

#     # for resnet20 on cifar10
#     # my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                         # milestones=[100, 150], last_epoch=-1)

#     # define learning rate schedule
#     if args.net_type == 'resnet':
#         if args.dataset == 'cifar10':
#             my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                         milestones=[100, 150], last_epoch=-1)
#         else:
#             decayRate = 0.97
#             my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

#     else:
#         decayRate = 0.97
#         my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


#     # set path for saving training logs
#     if not os.path.exists(os.path.join('runs', args.expname)):
#         os.makedirs(os.path.join('runs', args.expname))
#     log_path = os.path.join('runs', args.expname, 'training_log.csv')
#     with open(log_path, 'w') as f:
#         f.write('epoch,time(s),train_loss,test_loss,test_acc(%)\n')

#     cudnn.benchmark = True

#     # Define curriculum
#     epoch_curr = args.epochs
#     print('epochs: ', epoch_curr)
#     sigma_curr = args.sigmas
#     print('sigmas: ', sigma_curr)
#     kernel_curr = [6*e+1 for e in sigma_curr]
#     print('kernels: ', kernel_curr)
#     epoch_cumul_curr = np.cumsum(np.asarray(epoch_curr))
#     print(epoch_cumul_curr)
#     args.epochs = epoch_cumul_curr[-1]
    
#     # Freeze a random set of parameters from first n_freeze layers 
#     n_freeze = 3
#     p_freeze = 0.5
#     ct = 0
#     # for child in model.children():
#     # for child in list(list(model.children())[0].children())[0].children():
#     #     ct += 1
#     #     if ct < n_freeze:
#     #         for param in child.parameters():
#     #             freeze = np.random.choice(2, 1, p=[1-p_freeze, p_freeze])[0]
#     #             if freeze == 1:
#     #                 param.requires_grad = False

#     init_model_state = deepcopy(model.state_dict())

#     for epoch in range(0, epoch_cumul_curr[-1]):
        
#         if epoch == epoch_cumul_curr[-2]:
#             # Stochastic restore
#             restore_prob = 0.1
#             print(f"\nRestoring {restore_prob} weights to the initial weights")
#             for nm, m  in list(model.named_modules())[3:]:
#                 for npp, p in m.named_parameters():
#                     if npp in ['weight', 'bias'] and p.requires_grad:
#                         mask = (torch.rand(p.shape) < restore_prob).float().cuda() 
#                         with torch.no_grad():
#                             p.data = init_model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
                            
#             # evaluate on validation set
#             acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

#         # Pick kernel, sigma value for current epoch as per curriculum
#         idx = next(x for x, xval in enumerate(epoch_cumul_curr) if xval > epoch)
#         epoch_arr = np.asarray(epoch_curr[:idx+1])
#         epoch_cumul_arr = np.asarray(epoch_cumul_curr[:idx+1])
#         kernel_arr = np.asarray(kernel_curr[:idx+1])
#         sigma_arr = np.asarray(sigma_curr[:idx+1])
#         # weightage = epoch_arr/sum(epoch_arr)
#         weightage = epoch_cumul_arr/sum(epoch_cumul_arr)

#         print(f'Epoch {epoch}, kernel = {kernel_arr}, sigma = {sigma_arr}, weights = {weightage}')

#         # adjust_learning_rate(optimizer, epoch)

#         # train for one epoch
#         train_loss, epoch_time = train(train_loader, model, criterion, optimizer, epoch, kernel_arr, sigma_arr, weightage)

#         # evaluate on validation set
#         acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

#         # remember best prec@1 and save checkpoint
#         is_best = acc1 >= best_acc1
#         # print(f'best_acc: {best_acc1}, acc: {acc1}, is_best: {is_best}')
#         best_acc1 = max(acc1, best_acc1)
#         if is_best:
#             best_acc5 = acc5

#         with open(log_path, 'a') as f:
#             f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
#                 (epoch + 1),
#                 epoch_time,
#                 train_loss,
#                 val_loss,
#                 acc1,
#             ))

#         print(f'Val accuracy, current = {acc1}, best = {best_acc1}')        
            
#         save_checkpoint({
#             'epoch': epoch,
#             'arch': args.net_type,
#             'state_dict': model.state_dict(),
#             'best_acc1': best_acc1,
#             'best_acc5': best_acc5,
#             'optimizer': optimizer.state_dict(),
#         }, is_best)

#         my_lr_scheduler.step()

#     print('Best accuracy (top-1 and 5 acc):', best_acc1, best_acc5)


# class BlurMulti(object):
#     """Blur the image in a sample.
#     """

#     def __init__(self, kernel_arr, sigma_arr, prob_arr):
#         self.kernel_arr = kernel_arr
#         self.sigma_arr = sigma_arr
#         self.prob_arr = prob_arr

#     def __call__(self, sample):
#         # pick a blur level
#         indices_arr = np.arange(len(self.kernel_arr))
#         idx = np.random.choice(indices_arr, size=1, p=self.prob_arr)[0]
#         sigma = self.sigma_arr[idx].astype('float')  
#         kernel = int(self.kernel_arr[idx])

#         # apply blur
#         if sigma > 0:
#             sample = transforms.functional.gaussian_blur(sample, kernel, sigma) 

#         return sample


# def train(train_loader, model, criterion, optimizer, epoch, kernel_arr, sigma_arr, weightage):
#     # batch_time = AverageMeter()
#     # data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # print(f'Epoch {epoch}, Sigma {sigma_arr[-1]}, Kernel {kernel_arr[-1]}')

#     # switch to train mode
#     model.train()

#     # define blur transform
#     blur = BlurMulti(kernel_arr, sigma_arr, weightage)
#     # composed = transforms.Compose(blur)
                                
#     end = time.time()
#     current_LR = get_learning_rate(optimizer)[0]
#     for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#         # data_time.update(time.time() - end)

#         input = input.cuda()
#         target = target.cuda()

#         # blur input images
#         input = blur(input)

#         # r = np.random.rand(1)
#         # if args.beta > 0 and r < args.cutmix_prob:
#         #     # generate mixed sample
#         #     lam = np.random.beta(args.beta, args.beta)
#         #     rand_index = torch.randperm(input.size()[0]).cuda()
#         #     target_a = target
#         #     target_b = target[rand_index]
#         #     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
#         #     input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
#         #     # adjust lambda to exactly match pixel ratio
#         #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
#         #     # compute output
#         #     output = model(input)
#         #     loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
#         # else:
#         #     # compute output
#         #     output = model(input)
#         #     loss = criterion(output, target)

#         output = model(input)
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

#         losses.update(loss.item(), input.size(0))
#         top1.update(acc1.item(), input.size(0))
#         top5.update(acc5.item(), input.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if i % args.print_freq == 0 and args.verbose == True:
#             print('Epoch: [{0}/{1}][{2}/{3}]\t'
#                   'LR: {LR:.6f}\t'
#                 #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                 #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'.format(
#                 epoch, args.epochs, i, len(train_loader), LR=current_LR, loss=losses, top1=top1))

#     # measure elapsed time
#     epoch_time = time.time() - end

#     print('* Epoch: [{0}/{1}]\t Time {2}\t Top 1-acc {top1.avg:.3f}  \t Train Loss {loss.avg:.3f}'.format(
#         epoch, args.epochs, epoch_time, top1=top1, loss=losses))

#     # save blurred image for viewing
#     # print(input.cpu().shape)
#     # img = (input.detach().cpu().numpy())[0]
#     # img = np.swapaxes(img, 0, 2)
#     # print(img.shape)
#     # pilimg = Image.fromarray(np.uint8(img*255))
#     # pilimg.save(f'runs/{args.expname}/inp_ep{epoch}.png')

#     return losses.avg, epoch_time


# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2


# def validate(val_loader, model, criterion, epoch):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda()

#         output = model(input)
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

#         losses.update(loss.item(), input.size(0))

#         top1.update(acc1.item(), input.size(0))
#         top5.update(acc5.item(), input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0 and args.verbose == True:
#             print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'.format(
#                 #   'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
#                 epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
#                 top1=top1, top5=top5))

#     print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f} \t Test Loss {loss.avg:.3f}'.format(
#         epoch, args.epochs, top1=top1, loss=losses))
#     return top1.avg, top5.avg, losses.avg


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     directory = "runs/%s/" % (args.expname)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = directory + filename
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if args.dataset.startswith('cifar'):
#         lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
#     elif args.dataset == ('imagenet'):
#         if args.epochs == 300:
#             lr = args.lr * (0.1 ** (epoch // 75))
#         else:
#             lr = args.lr * (0.1 ** (epoch // 30))
#     elif args.dataset == ('tinyimagenet'):
#         lr =  args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def get_learning_rate(optimizer):
#     lr = []
#     for param_group in optimizer.param_groups:
#         lr += [param_group['lr']]
#     return lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#         # wrong_k = batch_size - correct_k
#         # res.append(wrong_k.mul_(100.0 / batch_size))
#         res.append(correct_k.mul_(100.0 / batch_size))

#     return res


# if __name__ == '__main__':
#     main()
