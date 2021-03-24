# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2
freeze2d = False

import torch, iou
import scannet_loader_2d3d as data
import torch.nn as nn
import torch.optim as optim
import sparseconvnet as scn
import time
import numpy as np
from unet2d.unet_model import UNet
from unet2d.unet_parts import SELayer

use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_notResidualBlocks'
pretrain2D='checkpoints/pretrain2D_epoch19.pth'
epoch_start = 3

class_weight = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837])
class_weight = (1-class_weight/class_weight.sum())**3
class_weight = class_weight/class_weight.max()
class_weight = torch.from_numpy(class_weight).float().cuda()



class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.numpoints = data.sample_points
        self.inputLayer = scn.InputLayer(data.dimension,data.full_scale, mode=4)
        self.submanifoldConvolution = scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)
        self.unet = scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)
        self.bn = scn.BatchNormReLU(m)
        self.output = scn.OutputLayer(data.dimension)

        self.net_2d = UNet(n_channels=3, n_classes=13, bilinear=True)
        self.linear_2d = nn.Linear(64, m)
        self.atten_2d = SELayer(64, reduction=4)

        self.linear = nn.Linear(m, 13)
    def forward(self,x,img,depth_p,valididx,sample_ind):
        x = self.inputLayer(x)
        x = self.submanifoldConvolution(x)
        x = self.unet(x)
        x = self.bn(x)
        x = self.output(x)

        f_2d, logist_2d = self.net_2d(img)

        logist_2d_pro = nn.functional.softmax(logist_2d.detach(), dim=1)
        logist_2d_pro = torch.clamp_min(logist_2d_pro, min=1e-12)
        entropy = torch.sum(logist_2d_pro * torch.log(logist_2d_pro), dim=1)
        entropy = torch.unsqueeze(entropy, 1)

        depth_p = torch.unsqueeze(depth_p, 1)

        f_2d_pro = f_2d.detach()

        f_2d_pro = entropy*depth_p*f_2d_pro

        f_2d_pro = self.atten_2d(f_2d_pro)


        n, c, h, w = f_2d_pro.shape
        f_2d_pro = f_2d_pro.view(n, c, h * w).permute(0, 2, 1).reshape(n * h * w, c)[valididx][sample_ind]
        f_2d_pro = self.linear_2d(f_2d_pro)

        x = x + f_2d_pro

        x=self.linear(x)
        return x,logist_2d

unet=Model()

training_epochs=500

weights = torch.load(pretrain2D)
net_dict = unet.net_2d.state_dict()
for k,v in weights.items():
    if k in net_dict:
        if v.shape == net_dict[k].shape:
            net_dict[k]=v
        else:
            print('%s shape not match' % (k))
    else:
        print('%s not in 2D model'%(k))
unet.net_2d.load_state_dict(net_dict)
print('load pretrain 2D model %s'%pretrain2D)


if use_cuda:
    unet=unet.cuda()

for param in unet.net_2d.parameters():
    param.requires_grad = False

params_3d = filter(lambda p: p.requires_grad, unet.parameters())
optimizer_3d = optim.Adam(params=params_3d,lr=0.001)
print('#classifer 3d parameters', sum([x.nelement() for x in [i for i in unet.parameters() if i.requires_grad==True]]))

for param in unet.net_2d.parameters():
    param.requires_grad = True

optimizer_2d = optim.SGD(params=unet.net_2d.parameters(),lr=0.0001,momentum=0.9, weight_decay=1e-4)
print('#classifer 2d parameters', sum([x.nelement() for x in unet.net_2d.parameters()]))

BEST_MOIU = 0
BEST_EPOCH = 0
for epoch in range(epoch_start,training_epochs):

    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss_3d=0
    train_loss_2d = 0
    for i,batch in enumerate(data.train_data_loader):
        torch.cuda.empty_cache()
        optimizer_3d.zero_grad()
        optimizer_2d.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
            batch['y']=batch['y'].cuda()
            batch['img'] = batch['img'].cuda()
            batch['label'] = batch['label'].cuda()
            batch['depth_p'] = batch['depth_p'].cuda()
        predict_3d,predict_2d = unet(batch['x'],batch['img'],batch['depth_p'],batch['valid_uv'],batch['sample_ind'])
        loss_3d = torch.nn.functional.cross_entropy(predict_3d,batch['y'],weight=class_weight)
        train_loss_3d+=loss_3d.item()
        loss_3d.backward(retain_graph=True)


        loss_2d = torch.nn.functional.cross_entropy(predict_2d, batch['label'], weight=class_weight)
        train_loss_2d += loss_2d.item()
        loss_2d.backward()

        optimizer_3d.step()
        optimizer_2d.step()

        lr = optimizer_2d.param_groups[0]['lr']
        print('[%d/%d] [%d/%d], loss_3d: %.4f, loss_2d: %.4f, lr: %.8f' % (epoch, training_epochs, i, len(data.train_data_loader), loss_3d.item(),loss_2d.item(), lr))

    if (epoch) %5 ==0:
        with torch.no_grad():
            unet.eval()
            store_pre_3d, = []
            store_gt_3d = []
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            start = time.time()
            for i, batch in enumerate(data.val_data_loader):
                if use_cuda:
                    batch['x'][1] = batch['x'][1].cuda()
                    batch['y'] = batch['y'].cuda()
                    batch['img'] = batch['img'].cuda()
                    batch['label'] = batch['label'].cuda()
                    batch['depth_p'] = batch['depth_p'].cuda()
                predict_3d, predict_2d = unet(batch['x'],batch['img'],batch['depth_p'],batch['valid_uv'],batch['sample_ind'])
                store_pre_3d.append(predict_3d.max(1)[1].data.cpu().numpy().astype(np.uint8))
                store_gt_3d.append(batch['y'].data.cpu().numpy().astype(np.uint8))


                print('[%d/%d]'%(i, len(data.val_data_loader)))
            '''--------------- evaluate 3d----------------'''
            print(epoch, 'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count / len(data.val_files) / 1e6,
                  'MegaHidden', scn.forward_pass_hidden_states / len(data.val_files) / 1e6, 'time=',
                  time.time() - start, 's')
            store_pre_3d = np.concatenate(store_pre_3d, axis=0)
            store_gt_3d = np.concatenate(store_gt_3d, axis=0)
            mean_iou, class_ious,mean_acc,class_acc,oAcc, CLASS_LABELS = iou.evaluate(store_pre_3d, store_gt_3d)

            if mean_iou > BEST_MOIU:
                BEST_MOIU = mean_iou
                BEST_EPOCH = epoch
                scn.checkpoint_save(unet, exp_name, 'unet', epoch, use_cuda)
            lr = optimizer_3d.param_groups[0]['lr']
            print("******** epoch: %d, miou: %.4f, best mIoU: %.4f, best epoch: %d ********\n" % (
                epoch, mean_iou, BEST_MOIU, BEST_EPOCH))
            with open('log_3d.txt', 'a+') as log_file:
                log_file.writelines("******** epoch: %d, miou: %.4f, best mIoU: %.4f, best epoch: %d, current_lr: %.8f ********\n" % (
                epoch, mean_iou, BEST_MOIU, BEST_EPOCH,lr))
                N_CLASSES = len(CLASS_LABELS)
                for i in range(N_CLASSES):
                    label_name = CLASS_LABELS[i]
                    log_file.writelines(
                        '{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}) \n'.format(label_name, class_ious[label_name][0],
                                                                            class_ious[label_name][1],
                                                                            class_ious[label_name][2]))
    if (epoch) % (training_epochs//5) == 0:
            for param in optimizer_2d.param_groups:
                param['lr'] *= 0.5