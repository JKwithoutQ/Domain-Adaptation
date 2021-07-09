import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
from data_loader import GetLoader
from model_attention import *
from model_dann import DANN_Resnet
import os
import torchvision.transforms as T

###
SRC_DIR = './datasets/office31/amazon'
TAR_DIR = './datasets/office31/dslr'
###
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LEARNING_RATE = 1e-2
BATCH_SIZE = 8
IMAGE_SIZE = 256
ITERNATION = 10000


# 导入数据
def load_train_data():
    img_transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # source
    src_dataset = torchvision.datasets.ImageFolder(
        root=SRC_DIR,
        transform=img_transform
    )
    src_dataloader = torch.utils.data.DataLoader(
        dataset=src_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    # target
    tar_dataset = torchvision.datasets.ImageFolder(
        root=TAR_DIR,
        transform=img_transform
    )
    tar_dataloader = torch.utils.data.DataLoader(
        dataset=tar_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    return src_dataloader, tar_dataloader


def load_test_data():
    img_transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # source
    src_dataset = torchvision.datasets.ImageFolder(
        root=SRC_DIR,
        transform=img_transform
    )
    src_dataloader = torch.utils.data.DataLoader(
        dataset=src_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    # target
    tar_dataset = torchvision.datasets.ImageFolder(
        root=TAR_DIR,
        transform=img_transform
    )
    tar_dataloader = torch.utils.data.DataLoader(
        dataset=tar_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    return src_dataloader, tar_dataloader


def exp_lr_scheduler(optimizer, step, init_lr=LEARNING_RATE, lr_decay_step=ITERNATION, step_decay_weight=0.08):
    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    # if step % 100 == 0:
    #     print('step is {} learning rate is set to {}'.format(step, current_lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


def test(step, net):
    src_loader, tar_loader = load_test_data()
    src_len, tar_len = len(src_loader), len(tar_loader)
    src_iter, tar_iter = iter(src_loader), iter(tar_loader)
    src_correct, tar_correct = 0, 0
    net.eval()
    with torch.no_grad():
        for img, lab in src_iter:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            rlt, _ = net(input_data=img)
            pred = torch.max(rlt, 1)
            src_correct += (pred[1] == lab).sum().item()
            print(pred[1].shape, lab.shape)
            print(pred[1], lab)

        for img, lab in tar_iter:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            rlt, _ = net(input_data=img)
            pred = torch.max(rlt, 1)
            tar_correct += (pred[1] == lab).sum().item()

    src_acc = src_correct / (src_len * BATCH_SIZE)
    tar_acc = tar_correct / (tar_len * BATCH_SIZE)
    line = 'Step:{} src_acc:{} tar_acc:{}'.format(step, src_acc, tar_acc)
    print(line)
    # epoch_rlt = open('./epoch_result.txt', 'a')
    # epoch_rlt.write(line)
    # epoch_rlt.close()
    return src_acc, tar_acc


def train(net, optimizer):
    print('train')
    # data
    iternation = 10000
    src_loader, tar_loader = load_train_data()
    src_len, tar_len = len(src_loader), len(tar_loader)
    src_iter, tar_iter = iter(src_loader), iter(tar_loader)
    # train
    for step in range(1, iternation):
        net.train()
        if step % src_len == 0:
            src_iter = iter(src_loader)
        if step % tar_len == 0:
            tar_iter = iter(tar_loader)
        # data
        src_img, src_lab = src_iter.next()
        tar_img, tar_lab = tar_iter.next()

        src_img, src_lab = src_img.to(DEVICE), src_lab.to(DEVICE)
        tar_img, tar_lab = tar_img.to(DEVICE), tar_lab.to(DEVICE)

        src_domain_lab = torch.zeros([BATCH_SIZE, 1]).float().to(DEVICE)
        tar_domain_lab = torch.ones([BATCH_SIZE, 1]).float().to(DEVICE)
        # train
        img = torch.cat([src_img, tar_img], dim=0)
        rlt, domain = net(input_data=img)

        src_rlt, tar_rlt = torch.chunk(rlt, 2, dim=0)
        src_domain, tar_domain = torch.chunk(domain, 2, dim=0)
        # loss
        # loss = F.cross_entropy(src_rlt, src_lab) + 0.5 * F.cross_entropy(src_domain, src_domain_lab) + 0.5 * F.cross_entropy(tar_domain, tar_domain_lab)
        loss = F.cross_entropy(src_rlt, src_lab) + 0.5 * F.binary_cross_entropy(src_domain, src_domain_lab) + 0.5 * F.binary_cross_entropy(tar_domain, tar_domain_lab)
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exp_lr_scheduler(optimizer, step)
        # test
        if step % 100 == 0:
            src_acc, tar_acc = test(step, net)
            # line = 'Step:{} src_acc:{} tar_acc:{}'.format(step, src_acc, tar_acc)
            # print(line)
            torch.save(net, './rlt_dann/dann_{}.pth'.format(step))


if __name__ == '__main__':
    torch.random.manual_seed(100)
    net = DANN_Resnet(n_class=31).to(DEVICE)
    # net = DANN_Resnet_attention(n_class=31).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
    print('DEVICE:', DEVICE)
    train(net, optimizer)
