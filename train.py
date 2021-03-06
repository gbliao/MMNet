import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyData
from MMNet import RGBres2net50,Depthres2net50,FusionNet
from functions import imsave
import argparse
from trainer import Trainer
import os

# same configuration
configurations = {
    1: dict(
        max_iteration=1200000,
        lr=1e-8,
        momentum=0.9,
        weight_decay=0.0005,
        spshot=200000,
        nclass=2,
        sshow=10,
    )
}

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--train_dataroot', type=str, default='../train_data_augment', help='path to train data')
parser.add_argument('--snapshot_root', type=str, default='./train_model', help='path to saved_models')
parser.add_argument('--salmap_root', type=str, default='./sal_map', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations[args.config]
cuda = torch.cuda.is_available()

"""""""""""~~~ dataset loader ~~~"""""""""
train_dataRoot = args.train_dataroot
if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)

if args.phase == 'train':
    SnapRoot = args.snapshot_root
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
print ('data already')

start_epoch = 0
start_iteration = 0

""""""""""" ~~~nets~~~ """""""""
model_rgb = RGBres2net50()
model_depth = Depthres2net50()
model_fusion = FusionNet()

if args.phase == 'train':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_rgb.load_state_dict(torch.load('./res2net50.pth', map_location=device))       # pre_trained_root
    model_depth.load_state_dict(torch.load('./res2net50.pth', map_location=device))     # pre_trained_root

if cuda:
    model_rgb = model_rgb.cuda()
    model_depth = model_depth.cuda()
    model_fusion = model_fusion.cuda()

if __name__ == '__main__':
    if args.phase == 'train':
        optimizer_rgb = optim.SGD(model_rgb.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        optimizer_depth = optim.SGD(model_depth.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        optimizer_fusion = optim.SGD(model_fusion.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fusion, step_size=35, gamma=0.1)

        training = Trainer(
            cuda=cuda,
            model_rgb=model_rgb,
            model_depth=model_depth,
            model_fusion=model_fusion,
            optimizer_rgb=optimizer_rgb,
            optimizer_depth=optimizer_depth,
            optimizer_fusion=optimizer_fusion,
            scheduler = scheduler,
            train_loader=train_loader,
            max_iter=cfg['max_iteration'],
            snapshot=cfg['spshot'],
            outpath=args.snapshot_root,
            sshow=cfg['sshow']
        )
        training.epoch = start_epoch
        training.iteration = start_iteration
        training.train()


