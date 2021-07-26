import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyTestData
from functions import imsave
import argparse
import os

from MMNet import RGBres2net50,Depthres2net50,FusionNet

# torch.set_num_threads(4)

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
parser.add_argument('--test_dataroot', type=str, default='./test_datasets', help='path to data')
parser.add_argument('--pre_trained_root', type=str, default='./train_model', help='path to pre trained')
args = parser.parse_args()
cuda = torch.cuda.is_available()

"""""""""""~~~ dataset loader ~~~"""""""""
print ('data already')

""""""""""" ~~~nets~~~ """""""""
model_rgb = RGBres2net50()
model_depth = Depthres2net50()
model_fusion = FusionNet()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.param is True:
    model_rgb.load_state_dict(torch.load(os.path.join(args.pre_trained_root, 'RGB.pth'),map_location=device))
    model_depth.load_state_dict(torch.load(os.path.join(args.pre_trained_root, 'Depth.pth'),map_location=device))
    model_fusion.load_state_dict(torch.load(os.path.join(args.pre_trained_root, 'fusion.pth'),map_location=device))

if __name__ == '__main__':
    if cuda:
        model_rgb = model_rgb.cuda()
        model_depth = model_depth.cuda()
        model_fusion = model_fusion.cuda()

    test_datasets = ['NLPR']

    for dataset in test_datasets:
        save_path = './test_Results/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        test_dataRoot = './test_datasets/' + dataset + '/'
        test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                                  batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        for id, (data, depth, img_name, img_size) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs = Variable(data).cuda()
                inputs_depth = Variable(depth).cuda()
            else:
                inputs = Variable(data)
                inputs_depth = Variable(depth)

            n, c, h, w = inputs.size()
            depth = inputs_depth.view(n, h, w, 1).repeat(1, 1, 1, c)
            depth = depth.transpose(3, 1)
            depth = depth.transpose(3, 2)

            R1,R2,R3,R4,R5 = model_rgb(inputs)
            D1,D2,D3,D4,D5 = model_depth(depth)
            outputs_all = model_fusion(R1,R2,R3,R4,R5,D1,D2,D3,D4,D5)
            outputs_all = F.softmax(outputs_all, dim=1)
            outputs1 = outputs_all[0][1]
            outputs = outputs1.cpu().data.resize_(h, w)

            imsave(os.path.join(save_path, img_name[0] + '.png'), outputs, img_size)

        print('The ' + dataset + ' testing process has finished!')
    print('The all testing process has finished!')
    




