from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

class THRCropTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x),self.transform(x),self.transform(x)]
class THRCropTransforms1:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x),self.transform2(x),self.transform1(x)]

class THRCropTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x),self.transform(x),self.transform(x)]
class THRCropTransforms2:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform2(x),self.transform1(x),self.transform2(x)]

class FourCropTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x),self.transform(x),self.transform(x),self.transform(x)]

class TwoCropTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x),self.transform(x)]
class TwoCropTransforms1:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x),self.transform2(x)]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x)


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            haha = correct[:k]
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_ratesss_100epoch1(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 25:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 45:
        lr = args.learning_rate * 0.01
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch1c(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 20:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 50:
        lr = args.learning_rate * 0.01
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch2(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 20:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 45:
        lr = args.learning_rate * 0.01
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch3(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 45:
        lr = args.learning_rate * 0.01
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch4(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 30:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * 15 / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch5(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 30:
        lr = args.learning_rate * 1
    elif epoch > 80:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch5(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 30:
        lr = args.learning_rate * 1
    elif epoch > 80:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch6(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch7(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 15:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 30:
        lr = args.learning_rate * 1
    elif epoch > 60:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_ratesss_100epoch8(args, optimizer, epoch):
#     epoch = epoch + 1
#     lr = args.learning_rate
#     if epoch <= 20:
#         eta_min = lr * (args.lr_decay_rate ** 3)
#         lr = eta_min + (lr - eta_min) * (
#                 1 + math.cos(math.pi * epoch / args.epochs)) / 2
#     elif epoch > 50:
#         lr = args.learning_rate * 0.01
#     else:
#         lr = args.learning_rate*0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_ratesss_100epoch8(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 30:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 55:
        lr = args.learning_rate * 0.01
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch9(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 20:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    elif epoch > 50:
        lr = args.learning_rate * 0.01
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 45:
        lr = args.learning_rate * 0.0005
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    elif epoch > 45:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02a(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 20:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    elif epoch > 45:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 45:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    elif epoch > 65:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_afgr(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 70:
        lr = args.learning_rate * 0.05
    elif epoch > 55:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_1(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 50:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 70:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_1a(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 45:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 85:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_1b(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 40:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 65:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_2(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 50:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 85:
        lr = args.learning_rate * 0.05
    elif epoch > 70:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_3(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 55:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 85:
        lr = args.learning_rate * 0.05
    elif epoch > 70:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_4(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 55:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 90:
        lr = args.learning_rate * 0.05
    elif epoch > 75:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_5(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 45:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    elif epoch > 65:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02b_6(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 45:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.0005
    elif epoch > 65:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02_1(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 45:
        lr = args.learning_rate * 0.005
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02_2(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.005
    elif epoch > 45:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02_3(args, optimizer, epoch): #ganjuebutaixing
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.001
    elif epoch > 45:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epochAID02_4(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 35:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 85:
        lr = args.learning_rate * 0.0005
    elif epoch > 70:
        lr = args.learning_rate * 0.005
    elif epoch > 45:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss_100epoch(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 20:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.0005
    elif epoch > 45:
        lr = args.learning_rate * 0.005
    else:
        lr = args.learning_rate*0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_learning_rate_re(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 30:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 80:
        lr = args.learning_rate * 0.01
    elif epoch > 55:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_AID(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 40:
        lr = args.learning_rate * 0.1
    elif epoch > 85:
        lr = args.learning_rate * 0.005
    elif epoch > 65:
        lr = args.learning_rate * 0.01
    else:
        lr = args.learning_rate*0.05
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_AID1(args, optimizer, epoch):
    # epoch = epoch + 1
    lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_AID2(args, optimizer, epoch):
    epoch = epoch + 1
    # lr = args.learning_rate
    if epoch <= 40:
        lr = args.learning_rate
    elif epoch > 80:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_ratesss(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 90:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 120:
        lr = args.learning_rate * 0.0001
    elif epoch > 96:
        lr = args.learning_rate * 0.05
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_ratess(args, optimizer, epoch):
    epoch = epoch + 1
    lr = args.learning_rate
    if epoch <= 80:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif epoch > 100:
        lr = args.learning_rate * 0.0001
    elif epoch > 86:
        lr = args.learning_rate * 0.01
    else:
        lr = args.learning_rate*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rates(args, optimizer, epoch):
    epoch = epoch + 1
    if epoch <= 6:
        lr = args.learning_rate * epoch / 6
    elif epoch > 180:
        lr = args.learning_rate * 0.0001
    elif epoch > 160:
        lr = args.learning_rate * 0.01
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    # a = model.parameters()
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer
def set_optimizer429(opt, parameters):
    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def save_model_regular(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state