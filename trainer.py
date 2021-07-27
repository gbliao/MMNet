import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

running_loss_final = 0
writer = SummaryWriter(comment='_comment', filename_suffix="_filename_suffix")

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    input = input.transpose(1,2).transpose(2,3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):
    def __init__(self, cuda, model_rgb,model_depth,model_fusion,
                 optimizer_rgb,optimizer_depth,optimizer_fusion,scheduler,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.model_fusion = model_fusion
        self.optim_rgb = optimizer_rgb
        self.optim_depth = optimizer_depth
        self.optimizer_fusion = optimizer_fusion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average

    def train_epoch(self):
        for batch_idx, (data, target, depth) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                data, target, depth = data.cuda(), target.cuda(), depth.cuda()
            data, target, depth = Variable(data), Variable(target), Variable(depth)
            n, c, h, w = data.size()
            depth = depth.view(n,h,w,1).repeat(1,1,1,c)
            depth = depth.transpose(3,1)
            depth = depth.transpose(3,2)
            self.optim_rgb.zero_grad()
            self.optim_depth.zero_grad()
            self.optimizer_fusion.zero_grad()
            global running_loss_final

            R1,R2,R3,R4,R5 = self.model_rgb(data)
            D1,D2,D3,D4,D5 = self.model_depth(depth)
            score_fusion = self.model_fusion(R1,R2,R3,R4,R5,D1,D2,D3,D4,D5)

            loss_all = cross_entropy2d(score_fusion, target, size_average=self.size_average)

            running_loss_final += loss_all.item()

            if iteration % 10 == (10 - 1):
                writer.add_scalar("Train Loss", running_loss_final / (n * self.sshow), iteration)
            if iteration % 10 == (10 - 1):
                print('\n [Epoch=%3d, iters=%6d, The training loss of Net: %.3f]' % (
                    self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow)))
            if iteration % self.sshow == (self.sshow-1):
                running_loss_final = 0.0

            if iteration <= 800000:
                if iteration % self.snapshot == (self.snapshot-1):
                    savename = ('%s/RGB_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (RGB: %d)' % (iteration+1))
                    savename_depth = ('%s/Depth_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (Depth: %d)' % (iteration+1))
                    savename_fusion = ('%s/fusion_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_clstm.state_dict(), savename_fusion)
                    print('save: (fusion: %d)' % (iteration+1))
            else:
                if iteration % 10000 == (10000 - 1):
                    savename = ('%s/RGB_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (RGB: %d)' % (iteration + 1))
                    savename_depth = ('%s/Depth_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (Depth: %d)' % (iteration + 1))
                    savename_fusion = ('%s/fusion_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_clstm.state_dict(), savename_fusion)
                    print('save: (fusion: %d)' % (iteration + 1))

                if (iteration+1) == self.max_iter:
                    savename = ('%s/RGB_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (RGB: %d)' % (iteration+1))
                    savename_depth = ('%s/Depth_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (Depth: %d)' % (iteration+1))
                    savename_fusion = ('%s/fusion_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_clstm.state_dict(), savename_fusion)
                    print('save: (fusion: %d)' % (iteration+1))

            loss_all.backward()
            self.optimizer_fusion.step()
            self.optim_depth.step()
            self.optim_rgb.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()
            if self.iteration >= self.max_iter:
                break
