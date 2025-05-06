"""
Load the pre-training weights
resnet 7x7 conv1
save the parameters
"""

from __future__ import print_function

import os
gpu_id='0,1,3'
os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
import sys
import argparse
import time
from data.ClassAwareSampler import get_sampler
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn.functional as F
from utils.CreateDataloader import LoadData
from utils.util import TwoCropTransform, TwoCropTransforms, AverageMeter, THRCropTransforms
from utils.util import adjust_learning_ratesss_100epochAID02, warmup_learning_rate
from utils.util import set_optimizer , save_model
from networks.resnet_big_v4 import SupConResNet_unite_new_classifer_difftrans427_no_linear
from utils.losses import SupConLoss
from utils.self_losses_temp import SupConLoss as self_supconloss
import utils.loader as loader


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--class_number', type=int, default=45,
                        help='the number of the classes')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--self_batch_size', type=int, default=128,
                        help='self batch_size')
    parser.add_argument('--classifier_batch_size', type=int, default=128,
                        help='classifier batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='./NWPU_RESISC45',
                        help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # long tail
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--max_num_per_class', default=600, type=float, help='max number')

    # temperature
    parser.add_argument('--temp', type=float, default=0.6,
                        help='temperature for loss function')
    parser.add_argument('--self_temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--ld', type=float, default=0.1,
                        help='lambda for adapter')

    # other setting
    parser.add_argument('--cosine', action='store_true',default=True,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)


    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    #
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = (0.36801905,0.3809775,0.34357441)
        std = (0.14530348,0.13557449,0.13204114)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(size=opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(size=opt.size),
        transforms.ToTensor(),
        normalize,
    ])



    if opt.dataset == 'path':
        train_dataset1 = LoadData("img_list_txt/NWPU_train_imb_0.01.txt", TwoCropTransforms(train_transform), True)
        train_dataset2 = LoadData("img_list_txt/NWPU_train_imb_0.01.txt", TwoCropTransform(transform_train), True)
        test_dataset = LoadData("img_list_txt/NWPU_test_imb_0.01.txt", TwoCropTransform(transform_test), True)
        self_con_dataset = LoadData("img_list_txt/self_con.txt", THRCropTransforms(train_transform), True)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    test_sampler = None
    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=opt.classifier_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=get_sampler()(train_dataset2, 4))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=(test_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=test_sampler)

    self_con_loader = torch.utils.data.DataLoader(
        self_con_dataset, batch_size=opt.self_batch_size, shuffle=(test_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=test_sampler)

    return train_loader1, train_loader2, self_con_loader, test_loader


def set_model(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weight_path = "backbone_weight/resnet50-pre.pth"
    model = SupConResNet_unite_new_classifer_difftrans427_no_linear(name=opt.model, classes_number=opt.class_number,ld = opt.ld)
    pre_weights = torch.load(model_weight_path, map_location=device)
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del pre_weights[key]
    model.encoder.load_state_dict(pre_weights, strict=False)
    SCcriterion = SupConLoss(temperature=opt.temp)
    self_SCcriterion = self_supconloss(temperature=opt.self_temp)
    CEcriterion1 = torch.nn.CrossEntropyLoss()
    CEcriterion2 = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
           model.encoder = torch.nn.DataParallel(model.encoder,device_ids=[0,1,2])
        model = model.to(device)

        CEcriterion1 = CEcriterion1.cuda()
        CEcriterion2 = CEcriterion2.cuda()
        SCcriterion = SCcriterion.cuda()
        self_SCcriterion = self_SCcriterion.cuda()
        cudnn.benchmark = True

    return model, SCcriterion, self_SCcriterion, CEcriterion1, CEcriterion2

def train(train_loader1, train_loader2, self_test_loader, model, SCcriterion, self_SCcriterion, CEcriterion1, CEcriterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    idx = 0
    alpha = (epoch / opt.epochs) #** 2
    for (images1, labels1), (images2, labels2),(images3,labels3) in zip(train_loader1, train_loader2, self_test_loader):
        data_time.update(time.time() - end)
        images3 = torch.cat([images3[0], images3[1], images3[2]], dim=0)
        if torch.cuda.is_available():
            images1_1 = images1[0].cuda(non_blocking=True)
            images1_2 = images1[1].cuda(non_blocking=True)
            # images1 = images1.cuda(non_blocking=True)
            labels1 = labels1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels2 = labels2.cuda(non_blocking=True)
            images3 = images3.cuda(non_blocking=True)
        bsz = labels1.shape[0]
        self_bsz = int(images3.shape[0] / 3)


        # compute loss
        features1, features2, targets1, targets2 = model(images1_1, images1_2, images2, images3)
        f1, f2, f3 = torch.split(features2, [self_bsz, self_bsz, self_bsz], dim=0)
        features2 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)


        m = model.prototypes.weight.data.clone()

        m = F.normalize(m, dim = 1)

        if opt.method == 'SupCon':
            loss = (1-alpha)*(0.3*self_SCcriterion(features2)+0.5*SCcriterion(features1, m, labels1) + 0.5*CEcriterion1(targets1, labels1)) + alpha*CEcriterion2(targets2,labels2)

        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        idx += 1
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss:.3f} ({loss:.3f})'.format(
                   epoch, idx + 1, len(train_loader1), batch_time=batch_time,
                   data_time=data_time, loss=loss))
            sys.stdout.flush()

    return losses.avg


def mytests(epoch, test_loader, model, loss, opt):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            a, b, c, outputs = model(images,images,images,images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    with open('result_acc/NWPU/no_linear_adapter_temp{}_imbfactor{}_bsz{}.txt'.format(opt.temp, opt.imb_factor, opt.batch_size),'a', encoding='UTF-8') as f:
        f.write('Epoch %d lamda %.2f Accuracy on test set: %.2f %%   loss %.4f\n' % (epoch, opt.ld ,100 * correct / total, loss))
    print('Accuracy on test set: %.2f %%    loss %.4f' % (100 * correct / total, loss))
    return 100 * correct / total

def main():
    opt = parse_option()

    # build data loader
    train_loader1, train_loader2, self_test_loader, test_loader = set_loader(opt)
    # build model and criterion
    model, SCcriterion, self_SCcriterion, CEcriterion1, CEcriterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    acc_list = []
    loss_list = []
    current = 0
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_ratesss_100epochAID02(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader1, train_loader2, self_test_loader, model, SCcriterion, self_SCcriterion, CEcriterion1, CEcriterion2,
                     optimizer, epoch, opt)
        accuracy = mytests(epoch, test_loader, model, loss, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        acc_list.append(accuracy)
        loss_list.append(loss)


        if accuracy > 98:
            if accuracy > current:
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
                current = accuracy

if __name__ == '__main__':
    main()