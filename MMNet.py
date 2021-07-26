from fusion import CMFM2,CMFM,FFU,MFU
from torch.autograd import Variable
import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
import os
import argparse
affine_par = True

class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, stride=1, bias=False) #
        self.bn1 = nn.BatchNorm2d(width*scale,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, dilation = dilation_, padding=dilation_, bias=False))
          bns.append(nn.BatchNorm2d(width, affine = affine_par))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        for j in range(self.nums):
            for i in self.bns[j].parameters():
                i.requires_grad = False
        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class RGBRes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(RGBRes2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation__ = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample,
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__, baseWidth = self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        R_nopool1 = x
        x = self.maxpool(x)
        R1 = R_nopool1
        x = self.layer1(x)
        R2 = x
        x = self.layer2(x)
        R3 = x
        x = self.layer3(x)
        R4 = x
        x = self.layer4(x)
        R5 = x
        return R1,R2,R3,R4,R5

class DepthRes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(DepthRes2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation__ = 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample,
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__, baseWidth = self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        h_nopool1 = x
        x = self.maxpool(x)
        D1 = h_nopool1
        x = self.layer1(x)
        D2 = x
        x = self.layer2(x)
        D3 = x
        x = self.layer3(x)
        D4 = x
        x = self.layer4(x)
        D5 = x
        return D1,D2,D3,D4,D5

def RGBres2net50():
    model = RGBRes2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 48, scale = 2)
    return model


def Depthres2net50():
    model = DepthRes2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 48, scale = 2)
    return model


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv_block2res1r = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block2res1d = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block3res1r = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block3res1d = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block4res1r = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1,bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block4res1d = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1,bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block5res1r = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block5res1d = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.self_ablock2 = CMFM2(in_channels=256)
        self.self_ablock3 = CMFM(in_dim=256,activation=True)
        self.self_ablock4 = CMFM(in_dim=256,activation=True)
        self.self_ablock5 = CMFM(in_dim=256,activation=True)
        self.conv_block2res2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block3res2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block4res2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.conv_block5res2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU())
        self.self_gau5_4 = FFU(64,64)
        self.self_gau4_3 = FFU(64,64)
        self.self_gau3_2 = FFU(64,64)
        self.self_match2 = MFU(64)
        self.self_match3 = MFU(64)
        self.self_match4 = MFU(64)
        self.self_match5 = MFU(64)
        self.conv_pred1 = nn.Conv2d(64*4, 64, 3, padding=1,bias=False)
        self.conv_pred2 = nn.Conv2d(64, 2, 3, padding=1,bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, R1,R2,R3,R4,R5,D1,D2,D3,D4,D5):
        r2 = self.conv_block2res1r(R2)
        r3 = self.conv_block3res1r(R3)
        r4 = self.conv_block4res1r(R4)
        r5 = self.conv_block5res1r(R5)
        d2 = self.conv_block2res1d(D2)
        d3 = self.conv_block3res1d(D3)
        d4 = self.conv_block4res1d(D4)
        d5 = self.conv_block5res1d(D5)

        gfmout_2 = self.self_ablock2(r2,d2)
        gfmout_3 = self.self_ablock3(r3,d3)
        gfmout_4 = self.self_ablock4(r4,d4)
        gfmout_5 = self.self_ablock5(r5,d5)

        gfmout_5 = self.conv_block5res2(gfmout_5)
        gfmout_4 = self.conv_block4res2(gfmout_4)
        gfmout_3 = self.conv_block3res2(gfmout_3)
        gfmout_2 = self.conv_block2res2(gfmout_2)

        grmout_5 = gfmout_5
        grmout_4 = self.self_gau5_4(grmout_5,gfmout_4)
        grmout_3 = self.self_gau4_3(grmout_4,gfmout_3)
        grmout_2 = self.self_gau3_2(grmout_3,gfmout_2)

        output2 = self.self_match2(grmout_2,skip_out=gfmout_2,side_out = None)
        output3 = self.self_match3(output2,skip_out=gfmout_3,side_out=grmout_3)
        output4 = self.self_match4(output3,skip_out=gfmout_4,side_out=grmout_4)
        output5 = self.self_match5(output4,skip_out=gfmout_5,side_out=grmout_5)

        output = torch.cat((
            F.interpolate(output2, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(output3, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(output4, scale_factor=16, mode='bilinear', align_corners=False),
            F.interpolate(output5, scale_factor=32, mode='bilinear', align_corners=False),
        ), 1)

        output = self.conv_pred1(output)
        output = self.conv_pred2(output)

        return output