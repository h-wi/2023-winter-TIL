# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

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
from resnet import resnet50_tutorial
from util import ColorJitter, Lighting
import numpy as np
import wandb
import warnings

from fvcore.nn import FlopCountAnalysis
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--projname', default='winter-tutorial-week3', type=str,
                    help='name of project')
parser.add_argument('--exp-name', default='default', type=str,
                    help='name of expriments')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='use wandb')
parser.add_argument("--cuda", type=str , default='0,1,2,3,4,5,6,7,8' , ##
                    help= 'CUDA VISIBLE_DEVICES options')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc1 = 0
best_acc5 = 0


def main():
    global args, best_acc1, best_acc5
    args = parser.parse_args()

    init_cuda_distributed(args)
    if args.log_wandb and args.is_master:
        wandb.init(project=args.projname,
                   name=args.exp_name,
                   config= {
                       "learning_rate": args.lr,
                       "momentum": args.momentum,
                       "batch_size": args.batch_size,
                       "epochs": args.epochs,
                       "dataset": args.dataset,
                       "cutmix_prob": args.cutmix_prob,
                       "device": args.cuda,
                        },
                   reinit=True,
               )

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
            train_sampler = DistributedSampler(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train), shuffle=True)
            test_sampler = DistributedSampler(
                datasets.CIFAR100('../data', train=False, download=True, transform=transform_train), shuffle=False)

            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                sampler=train_sampler, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                sampler=test_sampler, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

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

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    model = resnet50_tutorial()
    model.to(args.device)
    model = DDP(module=model, device_ids=[args.local_rank])

    if args.is_master:
        print(model)

    # To compute model_param
    param_count = sum([m.numel() for m in model.parameters()])
    print('the number of model parameters: {}'.format(param_count))

    example_input = torch.ones((args.batch_size, ) + (3, 32, 32), dtype=torch.float32)
    fca = FlopCountAnalysis(model, example_input.cuda())
    flops = fca.total()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc5 = acc5

        if args.is_master:
            print('Current best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_err1': best_acc1,
                'best_err5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        if args.log_wandb and args.is_master:
            wandb.log({"Model_param": param_count, "FLOPs": flops})
    if args.is_master:
        print('Best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    # To compute throughput
    t_start = time.perf_counter()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True and args.is_master:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    if args.is_master:
        print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    # throughput
    t_runtime = time.perf_counter() - t_start
    sample_per_sec = args.batch_size / t_runtime

    if args.log_wandb and args.is_master:
        wandb.log({"train-top1-acc": top1.avg, "train-top5-acc": top5.avg, "train-loss": losses.avg, "train-throughput": sample_per_sec})

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # To compute throughput
    t_start = time.perf_counter()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True and args.is_master:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    if args.is_master:
        print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    # throughput
    t_runtime = time.perf_counter() - t_start
    sample_per_sec = args.batch_size / t_runtime

    if args.log_wandb and args.is_master:
        wandb.log({"valid-top1-acc": top1.avg, "valid-top5-acc": top5.avg, "valid-loss": losses.avg, "valid-throughput": sample_per_sec})

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.exp_name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, reduce_every_n_step=50, reduce_on_compute=True):
        self.dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if self.dist:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.reduce_every_n_step = reduce_every_n_step
            self.reduce_on_compute = reduce_on_compute
        else:
            self.world_size = None
            self.reduce_on_compute = self.reduce_every_n_step = False

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

        if self.reduce_every_n_step and (self.count % self.reduce_every_n_step == 0):
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.count

def all_reduce_mean(val, wolrd_size):
    val = torch.Tensor([val]).to(args.device)
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    return float(val / wolrd_size)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # k만큼, 1차원, 큰 것부터, 정렬
    pred = pred.t() # transpose 2D, (batch_size, 1) -> (1 , batch_size)
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum(-1, keepdim=True)
        res.append(max(correct_k.mul_(100.0 / batch_size)))

    return res

def init_cuda_distributed(args): ##
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    args.distributed = True
    torch.distributed.init_process_group( backend='nccl', init_method='env://')

    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.local_world_size = int(os.environ['WORLD_SIZE'])

    args.is_master = args.local_rank == 0
    torch.cuda.set_device(args.local_rank) # 범인
    args.device = torch.device(f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    main()