import torch
import numpy as np
import cv2,os,glob
import sys
from plyfile import PlyData,PlyElement
sys.path.append('../')
from map import map2D_3D

import math,scipy,time
scale=50
elastic_deformation=False
ratio = 0.25
val_reps = 1
dimension=3
full_scale=4096
batch_size = 16
sample_points=4096
data_root_3d = '/media/biolab/Elements/S3DIS/sequence2'
data_root_2d = '/media/biolab/Elements/S3DIS/sequence'

label_to_names = {0: 'ceiling',1: 'floor',2: 'wall',3: 'beam',4: 'column',5: 'window',6: 'door',7: 'table',8: 'chair',
                9: 'sofa',10: 'bookcase',11: 'board',12: 'clutter'}

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag
def depth_probility(depth):
    # 计算均值图像和均值图像的平方图像
    depth_blur = cv2.blur(depth, (5, 5))
    reslut_1 = depth_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = depth ** 2
    reslut_2 = cv2.blur(img_2, (5, 5))

    # 计算图像的标准差
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
def get_train_filenames():
    files = []
    for area in [1, 2, 3, 4, 5, 6]:
        fs = glob.glob(os.path.join(data_root_3d, 'Area_%d' % area, '*/*.ply'))
        if area==5:
            fs=fs[::100]
        files.extend(fs)
    return files
def get_val_filenames():
    files = []
    for area in [5]:
        fs = glob.glob(os.path.join(data_root_3d, 'Area_%d' % area, '*/*.ply'))
        files.extend(fs)
    return files
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

def trainMerge(tbl):
    locs = []
    feats = []
    labels = []
    Sample_ind = []
    Valid_uv = []

    Imgs = []
    Labs = []
    Depth_p = []
    last_idx = 0
    for idx, i in enumerate(tbl):

        points, valid_uv, sample_ind = read_ply(train_files[i])

        a = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0)).astype(np.float32)
        b = (np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1)
        c = points[:, -1].astype(np.int32)

        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)
        if elastic_deformation:
            a = elastic(a, 6 * scale // 50, 40 * scale / 50)
            a = elastic(a, 20 * scale // 50, 160 * scale / 50)
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
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b) + torch.randn(3) * 0.1)
        labels.append(torch.from_numpy(c))

        Valid_uv.append(valid_uv)
        Sample_ind.append(sample_ind+last_idx)
        last_idx += np.sum(valid_uv)


        spt = train_files[i].replace("sequence2","sequence").replace(".ply",".png").replace("_process","").split('/')
        img_path = "/".join(spt[:-1]+['rgb']+spt[-1:])
        lbl_path = img_path.replace('rgb', 'semantic')
        depth_path = img_path.replace('rgb', 'depth')

        image = cv2.imread(img_path)
        img = image.astype(np.float32)
        img = img / 127.5 - 1
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()
        Imgs.append(img)

        label = cv2.imread(lbl_path, -1)
        label = label.astype(int)
        label = torch.from_numpy(label).long()
        Labs.append(label)

        depth = cv2.imread(depth_path, -1)
        depth = depth.astype(np.float32)
        depth /= 512.

        depth_p = depth_probility(depth)
        depth_p = torch.from_numpy(depth_p).float()
        Depth_p.append(depth_p)

    locs = torch.cat(locs, 0).long()
    feats = torch.cat(feats, 0).float()
    labels = torch.cat(labels, 0).long()
    Valid_uv = np.concatenate(Valid_uv, axis=0)
    Sample_ind = np.concatenate(Sample_ind, axis=0)


    Imgs = torch.stack(Imgs, 0).float()
    Labs = torch.stack(Labs, 0).long()
    Depth_p = torch.stack(Depth_p, 0).float()

    return {'x': [locs, feats], 'y': labels,'img':Imgs,'label':Labs,'depth_p':Depth_p,'valid_uv':Valid_uv,'sample_ind':Sample_ind, 'id': tbl}

def valMerge(tbl):
    locs = []
    feats = []
    labels = []
    Sample_ind = []
    Valid_uv = []

    Imgs = []
    Labs = []
    Depth_p = []
    last_idx = 0
    for idx, i in enumerate(tbl):
        points, valid_uv, sample_ind = read_ply(val_files[i])

        a = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0)).astype(np.float32)
        b = (np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1)
        c = points[:, -1].astype(np.int32)

        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)
        if elastic_deformation:
            a = elastic(a, 6 * scale // 50, 40 * scale / 50)
            a = elastic(a, 20 * scale // 50, 160 * scale / 50)
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
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))

        Valid_uv.append(valid_uv)
        Sample_ind.append(sample_ind + last_idx)
        last_idx += np.sum(valid_uv)

        spt = val_files[i].replace("sequence2", "sequence").replace(".ply", ".png").replace("_process","").split('/')
        img_path = "/".join(spt[:-1] + ['rgb'] + spt[-1:])
        lbl_path = img_path.replace('rgb', 'semantic')
        depth_path = img_path.replace('rgb', 'depth')

        image = cv2.imread(img_path)
        img = image.astype(np.float32)
        img = img / 127.5 - 1
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()
        Imgs.append(img)

        label = cv2.imread(lbl_path, -1)
        label = label.astype(int)
        label = torch.from_numpy(label).long()
        Labs.append(label)

        depth = cv2.imread(depth_path, -1)
        depth = depth.astype(np.float32)
        depth /= 512.
        depth_p = depth_probility(depth)
        depth_p = torch.from_numpy(depth_p).float()
        Depth_p.append(depth_p)
    locs = torch.cat(locs, 0).long()
    feats = torch.cat(feats, 0).float()
    labels = torch.cat(labels, 0).long()
    Valid_uv = np.concatenate(Valid_uv, axis=0)
    Sample_ind = np.concatenate(Sample_ind, axis=0)

    Imgs = torch.stack(Imgs, 0).float()
    Labs = torch.stack(Labs, 0).long()
    Depth_p = torch.stack(Depth_p, 0).float()
    return {'x': [locs, feats], 'y': labels, 'img': Imgs, 'label': Labs, 'depth_p': Depth_p, 'valid_uv': Valid_uv,
            'sample_ind': Sample_ind, 'id': tbl}

train_files = get_train_filenames()
val_files = get_val_filenames()

train_data_loader = torch.utils.data.DataLoader(
    list(range(len(train_files))),
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val_files))),
    batch_size=batch_size,
    collate_fn=valMerge,
    num_workers=16,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

if __name__=='__main__':
    for i, batch in enumerate(train_data_loader):
        print("*** [%d/%d] ***"%(i,len(train_data_loader)))
        print(batch['x'][0].shape)
        print(batch['x'][1].shape)
        print(batch['y'].shape)
        print(np.sum(batch['valid_uv'].shape))
        print(np.sum(batch['sample_ind']))
        print(batch['img'].shape)
        print(batch['label'].shape)
        print(batch['depth_p'].shape)


