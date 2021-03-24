# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

import torch,os,glob,math
from plyfile import PlyData,PlyElement
import numpy as np
# import scannet_2d3d_test as data
import torch.nn as nn
import sparseconvnet as scn
import iou
import cv2
# import time
# import numpy as np
from unet2d.unet_model import UNet
from unet2d.unet_parts import SELayer

# from map import write_ply_rgb,read_ply,id20_color
# from sklearn.neighbors import KDTree
scale=50
full_scale=4096
sample_points = 4096
dimension = 3
'''====================================== [ data ] ======================================='''
area = 'Area_5'

val_ply_root = '/media/biolab/Elements/S3DIS/sequence2/%s'%area
ply_rooms = [os.path.join(val_ply_root,i) for i in os.listdir(val_ply_root)]

val_pth_root = '/home/biolab/research/ASIS/data/stanford_indoor3d_pth'
pth_rooms = [os.path.join(val_pth_root,i) for i in os.listdir(val_pth_root)]

def get_frames(ply_rooms):
    file_list = glob.glob("{}/*.ply".format(ply_rooms))
    cameras = list(set([i.split('/')[-1].split('_')[1] for i in file_list]))
    cameras = sorted(cameras,key=lambda x:int(x))
    order_files = []
    for cam in cameras:
        temp = []
        for j in file_list:
            if j.split('/')[-1].split('_')[1]==cam:
                temp.append(j)
        temp = sorted(temp,key=lambda x:int(x.split('/')[-1].split('_')[2]))
        order_files.extend(temp)
    return order_files
def depth_probility(depth):

    depth_blur = cv2.blur(depth, (5, 5))
    reslut_1 = depth_blur ** 2

    img_2 = depth ** 2
    reslut_2 = cv2.blur(img_2, (5, 5))

    reslut = np.sqrt(np.maximum(reslut_2 - reslut_1, 0))

    reslut = maxpooling(reslut)
    reslut = maxpooling(reslut)
    reslut = maxpooling(reslut)

    reslut = reslut/np.max(reslut)
    reslut = 1-reslut
    return reslut

def maxpooling(img):
    ksize = 5
    img_left1 = np.zeros_like(img)
    img_left1[:, 1:] = img[:, :-1]
    img_left2 = np.zeros_like(img)
    img_left2[:, 2:] = img[:, :-2]

    img_right1 = np.zeros_like(img)
    img_right1[:, :-1] = img[:, 1:]
    img_right2 = np.zeros_like(img)
    img_right2[:, :-2] = img[:, 2:]

    img_up1 = np.zeros_like(img)
    img_up1[1:, :] = img[:-1, :]
    img_up2 = np.zeros_like(img)
    img_up2[2:, :] = img[:-2, :]

    img_down1 = np.zeros_like(img)
    img_down1[:-1, :] = img[1:, :]
    img_down2 = np.zeros_like(img)
    img_down2[:-2, :] = img[2:, :]

    fusion = np.stack([img, img_left1, img_left2, img_right1, img_right2, img_up1, img_up2, img_down1, img_down2],
                      axis=0)
    fusion = np.max(fusion, axis=0)
    return fusion
def read_ply(plypath):
    f = PlyData.read(plypath)
    data = f.elements[0].data
    x = np.array(data['x']).reshape(-1,1)
    y = np.array(data['y']).reshape(-1,1)
    z = np.array(data['z']).reshape(-1,1)
    r = np.array(data['red']).reshape(-1,1)
    g = np.array(data['green']).reshape(-1,1)
    b = np.array(data['blue']).reshape(-1,1)

    label = f.elements[1].data
    label = np.array(label['label']).reshape(-1,1)

    valid_uv = f.elements[2].data
    valid_uv = np.array(valid_uv['valid_uv']).reshape(-1).astype(np.bool)

    sample_ind = f.elements[3].data
    sample_ind = np.array(sample_ind['sample_ind']).reshape(-1)

    points = np.concatenate([x,y,z,r,g,b,label],axis=1)
    return points,valid_uv,sample_ind

def voxel_process(ply_path):
    points, valid_uv, sample_ind = read_ply(ply_path)

    a = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0)).astype(np.float32)
    b = (np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1)
    c = points[:, -1].astype(np.int32)

    m = np.eye(3) + np.random.randn(3, 3) * 0.1
    m[0][0] *= np.random.randint(0, 2) * 2 - 1
    m *= scale
    theta = np.random.rand() * 2 * math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    a = np.matmul(a, m)
    m = a.min(0)
    M = a.max(0)
    q = M - m
    offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + np.clip(
        full_scale - M + m + 0.001, None, 0) * np.random.rand(3)
    a += offset
    idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
    a = a[idxs]
    b = b[idxs]
    c = c[idxs]
    a = torch.from_numpy(a).long()
    locs = torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(0)], 1).long()
    feats = (torch.from_numpy(b)).float()
    labels = torch.from_numpy(c).long()

    Valid_uv = valid_uv
    Sample_ind = sample_ind

    # ------------------  2D ---------------------------
    spt = ply_path.replace("sequence2", "sequence").replace(".ply", ".png").replace("_process", "").split('/')
    img_path = "/".join(spt[:-1] + ['rgb'] + spt[-1:])
    lbl_path = img_path.replace('rgb', 'semantic')
    depth_path = img_path.replace('rgb', 'depth')

    image = cv2.imread(img_path)
    img = image.astype(np.float32)
    img = img / 127.5 - 1
    img = img.transpose(2, 0, 1)  # NHWC -> NCHW
    img = torch.from_numpy(img).float()
    Imgs = torch.unsqueeze(img,dim=0)

    label = cv2.imread(lbl_path, -1)
    label = label.astype(int)
    label = torch.from_numpy(label).long()
    Labs  = torch.unsqueeze(label,dim=0)

    depth = cv2.imread(depth_path, -1)
    depth = depth.astype(np.float32)
    depth /= 512.
    depth_p = depth_probility(depth)
    depth_p = torch.from_numpy(depth_p).float()
    Depth_p = torch.unsqueeze(depth_p,dim=0)
    return {'points':points,'x': [locs, feats], 'y': labels,'img':Imgs,'label':Labs,'depth_p':Depth_p,'valid_uv':Valid_uv,'sample_ind':Sample_ind}

'''====================================== [ model ] ======================================='''
use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_notResidualBlocks'
pretrainFusion = 'checkpoints/unet_scale20_m16_rep1_notResidualBlocks-000000032-unet.pth'


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.numpoints = sample_points
        self.inputLayer = scn.InputLayer(dimension,full_scale, mode=4)
        self.submanifoldConvolution = scn.SubmanifoldConvolution(dimension, 3, m, 3, False)
        self.unet = scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)
        self.bn = scn.BatchNormReLU(m)
        self.output = scn.OutputLayer(dimension)

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

weights = torch.load(pretrainFusion)
unet.load_state_dict(weights)

if use_cuda:
    unet=unet.cuda()
unet.eval()

'''====================================== [ evaluation ] ======================================='''
store_pre = []
store_gt = []
for i,ply_room in enumerate(ply_rooms):
    frames = get_frames(ply_room)
    for j,frame in enumerate(frames):
        batch = voxel_process(frame)
        points = batch['points']
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
            batch['img'] = batch['img'].cuda()
            batch['label'] = batch['label'].cuda()
            batch['depth_p'] = batch['depth_p'].cuda()
        predictions,_ = unet(batch['x'],batch['img'],batch['depth_p'],batch['valid_uv'],batch['sample_ind'])
        print("[%d/%d] [%d/%d]"%(i,len(ply_rooms),j,len(frames)))
        store_pre.append(predictions.max(1)[1].data.cpu().numpy())
        store_gt.append(batch['y'].data.cpu().numpy())
store_pre = np.concatenate(store_pre, axis=0)
store_gt = np.concatenate(store_gt, axis=0)
mean_iou, class_ious, mean_acc, class_acc, oAcc, CLASS_LABELS = iou.evaluate(store_pre, store_gt)
with open('%s.txt'%area, 'w') as log_file:
    log_file.writelines("mAcc: %.4f, oAcc: %.4f, miou: %.4f \n" % (mean_acc, oAcc, mean_iou))
    log_file.writelines("----------------- Acc ------------\n")
    N_CLASSES = len(CLASS_LABELS)
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        log_file.writelines(
            '{0:<14s}: {1:>5.3f}\n'.format(label_name, class_acc[label_name]))
    log_file.writelines("----------------- iou ------------\n")
    N_CLASSES = len(CLASS_LABELS)
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        log_file.writelines(
            '{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}) \n'.format(label_name, class_ious[label_name][0],
                                                                class_ious[label_name][1],
                                                                class_ious[label_name][2]))