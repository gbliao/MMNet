import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-5)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CA(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class sU(nn.Module):
    def __init__(self, out_channels):
        super(sU, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x = torch.sigmoid(self.conv(x))
        return x

class cU(nn.Module):
    def __init__(self, out_channels):
        super(cU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0,bias=False)
        self.conv2 = nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0,bias=False)
    def forward(self,x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class CMFM2(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(CMFM2, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2

        self.rgb1_conv1k = nn.Conv2d(self.in_channels, self.in_channels // 2, (1, self.kernel_size), padding=(0, pad))
        self.rgb1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.rgb1_convk1 = nn.Conv2d(self.in_channels // 2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.rgb1_bn2 = nn.BatchNorm2d(1)
        self.rgb2_convk1 = nn.Conv2d(self.in_channels, self.in_channels // 2, (self.kernel_size, 1), padding=(pad, 0))
        self.rgb2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.rgb2_conv1k = nn.Conv2d(self.in_channels // 2, 1, (1, self.kernel_size), padding=(0, pad))
        self.rgb2_bn2 = nn.BatchNorm2d(1)

        self.dep1_conv1k = nn.Conv2d(self.in_channels, self.in_channels // 2, (1, self.kernel_size), padding=(0, pad))
        self.dep1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.dep1_convk1 = nn.Conv2d(self.in_channels // 2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.dep1_bn2 = nn.BatchNorm2d(1)
        self.dep2_convk1 = nn.Conv2d(self.in_channels, self.in_channels // 2, (self.kernel_size, 1), padding=(pad, 0))
        self.dep2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.dep2_conv1k = nn.Conv2d(self.in_channels // 2, 1, (1, self.kernel_size), padding=(0, pad))
        self.dep2_bn2 = nn.BatchNorm2d(1)

        self.r_t_dcat_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.r_tdadd_cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.t_dcat_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.out_conv1 = nn.Conv2d(in_channels, in_channels// 2, kernel_size=1,  bias=False)
        self.out_conv2 = nn.Conv2d(in_channels, in_channels// 2, kernel_size=1,  bias=False)
        self.out_conv3 = nn.Conv2d(in_channels, in_channels// 2, kernel_size=1,  bias=False)
        self.output_ca = CA(in_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels*3//2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))

    def forward(self, xr , xd):
        rgb1_feats = F.relu(self.rgb1_bn1(self.rgb1_conv1k(xr)))
        rgb1_feats = F.relu(self.rgb1_bn2(self.rgb1_convk1(rgb1_feats)))
        rgb2_feats = F.relu(self.rgb2_bn1(self.rgb2_convk1(xr)))
        rgb2_feats = F.relu(self.rgb2_bn2(self.rgb2_conv1k(rgb2_feats)))
        rgb_feats = torch.sigmoid(torch.add(rgb1_feats, rgb2_feats))
        rgb_feats = rgb_feats.expand_as(xr).clone()

        dep1_feats = F.relu(self.dep1_bn1(self.dep1_conv1k(xd)))
        dep1_feats = F.relu(self.dep1_bn2(self.dep1_convk1(dep1_feats)))
        dep2_feats = F.relu(self.dep2_bn1(self.dep2_convk1(xd)))
        dep2_feats = F.relu(self.dep2_bn2(self.dep2_conv1k(dep2_feats)))
        dep_feats = torch.sigmoid(torch.add(dep1_feats, dep2_feats))
        dep_feats = dep_feats.expand_as(xd).clone()

        rd1_feats = torch.mul((1+rgb1_feats),dep1_feats)
        rd2_feats = torch.mul((1+rgb2_feats),dep2_feats)
        rd_feats = torch.sigmoid(torch.add(rd1_feats, rd2_feats))
        rd_feats = rd_feats.expand_as(xd).clone()

        out_r = torch.mul(rgb_feats,xr)
        out_d = torch.mul(dep_feats,xd)
        out_rd = torch.mul(rd_feats,xd)

        r_t_dout = out_rd+out_r+out_d
        r_t_dout = self.r_t_dcat_conv(r_t_dout)
        adout_rd = out_rd + out_d
        r_tdout = torch.mul(adout_rd, out_r)
        r_tdout = self.r_tdadd_cat_conv(r_tdout)
        t_dout = torch.mul(out_rd, out_d)
        t_dout = self.t_dcat_conv(t_dout)
        r_tdout =  self.out_conv1(r_tdout)
        t_dout =   self.out_conv2(t_dout)
        r_t_dout = self.out_conv3(r_t_dout)
        out = torch.cat([r_tdout, t_dout, r_t_dout], dim=1)
        out = self.out_conv(out)
        out = self.output_ca(out) * out
        return out

class CMFM(nn.Module):
    def __init__(self,activation,in_dim=2048):
        super(CMFM, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim
        self.activation = activation

        self.query_convr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.key_convr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.value_convr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.query_convd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.key_convd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.value_convd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.key_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 8, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.r_t_dcat_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.r_tdadd_cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.t_dcat_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))

        self.output_ca = CA(input_dim)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim*3, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))

        self.gamma_r = nn.Parameter(torch.zeros(1))
        self.gamma_d = nn.Parameter(torch.zeros(1))
        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xr, xd):
        m_batchsize, C, width, height = xr.size()

        query_r = self.query_convr(xr).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key_r = self.key_convr(xr).view(m_batchsize, -1, width * height)
        value_r = self.value_convr(xr).view(m_batchsize, -1, width * height)
        attention_r = self.softmax(torch.bmm(query_r, key_r))
        out_r = torch.bmm(value_r, attention_r.permute(0, 2, 1))
        out_r = out_r.view(m_batchsize, C, width, height)

        query_d = self.query_convd(xd).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key_d = self.key_convd(xd).view(m_batchsize, -1, width * height)
        value_d = self.value_convd(xd).view(m_batchsize, -1, width * height)
        attention_d = self.softmax(torch.bmm(query_d, key_d))
        out_d = torch.bmm(value_d, attention_d.permute(0, 2, 1))
        out_d = out_d.view(m_batchsize, C, width, height)

        query_guiderd = self.query_convrd(out_r).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key_guiderd = self.key_convrd(out_r).view(m_batchsize, -1, width * height)
        query_rd = torch.mul((1+query_guiderd),query_d)
        key_rd = torch.mul((1+key_guiderd),key_d)
        energy = torch.bmm(query_rd, key_rd)
        attention_rd = self.softmax(energy)
        value_rd = value_d
        out_rd = torch.bmm(value_rd, attention_rd.permute(0, 2, 1))
        out_rd = out_rd.view(m_batchsize, C, width, height)

        out_rd = self.gamma_rd * out_rd + xd
        out_r = self.gamma_r * out_r + xr
        out_d = self.gamma_d * out_d + xd

        r_t_dout = out_rd + out_r + out_d
        r_t_dout = self.r_t_dcat_conv(r_t_dout)
        adout_rd = out_rd + out_d
        r_tdout = torch.mul(adout_rd, out_r)
        r_tdout = self.r_tdadd_cat_conv(r_tdout)
        t_dout = torch.mul(out_rd, out_d)
        t_dout = self.t_dcat_conv(t_dout)
        out = torch.cat([r_tdout,t_dout,r_t_dout],dim=1)
        out = self.out_conv(out)
        out = self.output_ca(out)*out
        return out

class FFU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(FFU, self).__init__()
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low,eps=1e-05, momentum=0.1, affine=True)
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.smooth = nn.Conv2d(channels_high, channels_low, kernel_size=3,padding=1,bias=False)
        self.bn_upsample = nn.Sequential(
            nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_low, eps=1e-05, momentum=0.1, affine=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_high, f_low):
        b, c, h, w = f_high.shape
        f_high_gp = nn.AvgPool2d(f_high.shape[2:])(f_high).view(len(f_high), c, 1, 1)
        f_high_gp = self.conv1x1(f_high_gp)
        f_high_gp = self.relu(f_high_gp)
        f_low_mask = self.conv3x3(f_low)
        f_low_mask = self.bn_low(f_low_mask)
        f_att = f_low_mask * f_high_gp
        out = self.smooth(f_high)
        out = F.interpolate(out,scale_factor=2, mode='bilinear', align_corners=True) + f_att
        out = self.relu(self.bn_upsample(out))
        return out

class MFU(nn.Module):
    def __init__(self, out_channels):
        super(MFU, self).__init__()
        self.convstr = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.ReLU(inplace=True))
        self.spatial_gate = sU(out_channels)
        self.channel_gate = cU(out_channels)

    def forward(self,x,skip_out,side_out=None):
        skip_out = F.relu(self.conv1x1(skip_out))
        if side_out is not None:
            x = self.convstr(x)
            x = x + side_out + skip_out
        else :
            x = x + skip_out
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        out = g1*x + g2*x
        return out