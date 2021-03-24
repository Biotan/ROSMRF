# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2
import open3d as o3d
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
from sklearn.neighbors import KDTree
scale=50
full_scale=4096
sample_points = 4096
dimension = 3
'''====================================== [ data ] ======================================='''
area = 'Area_5'
NUM_CLASS=13

val_ply_root = '/media/biolab/SIM2/S3DIS/sequence2/%s'%area
ply_rooms = [os.path.join(val_ply_root,i) for i in os.listdir(val_ply_root)]

def get_frames(ply_rooms):
    file_list = glob.glob("{}/*.ply".format(ply_rooms))
    return file_list
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
    # labels = torch.from_numpy(c).long()
    labels = c

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

    depth = cv2.imread(depth_path, -1)
    depth = depth.astype(np.float32)
    depth /= 512.
    depth_p = depth_probility(depth)
    depth_p = torch.from_numpy(depth_p).float()
    Depth_p = torch.unsqueeze(depth_p,dim=0)
    return {'points':points,'x': [locs, feats], 'y': labels,'img':Imgs,'depth_p':Depth_p,'valid_uv':Valid_uv,'sample_ind':Sample_ind}

g_class2color = np.array([[0,255,0],
                         [0,0,255],
                         [0,255,255],
                         [200,200,100],
                         [255,0,255],
                         [100,100,255],
                         [255,255,0],
                         [170,120,200],
                         [255,0,0],
                         [200,100,100],
                         [10,200,100],
                         [200,200,200],
                         [50,50,50]])

def write_ply(save_path, points, text=False):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,6) [x,y,z,r,g,b,l]
    """
    vertex = list(map(lambda x: tuple(x), points[:,:6].tolist()))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'uint8'), ('green', 'uint8'),
                                     ('blue', 'uint8')])
    ver_el = PlyElement.describe(vertex, 'vertex')

    label = np.array(points[:,-1],dtype=[('label','uint8')])
    label = PlyElement.describe(label, 'semantic')

    PlyData([ver_el,label], text=text).write(save_path)


def open3d_process(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # "Downsample the point cloud with a voxel of 0.003"
    downpcd = pcd.voxel_down_sample(voxel_size=0.025)

    # "Statistical oulier removal"
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
    downpcd = downpcd.select_down_sample(ind)

    tree = KDTree(points[:, :3], leaf_size=50)
    sub_cloud = np.asarray(downpcd.points)
    dis, ind = tree.query(sub_cloud, k=1)

    color = points[ind[:, 0], 3:6]

    points = np.c_[sub_cloud, color]

    return points

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

        self.net_2d = UNet(n_channels=3, n_classes=NUM_CLASS, bilinear=True)
        self.linear_2d = nn.Linear(64, m)
        self.atten_2d = SELayer(64, reduction=4)

        self.linear = nn.Linear(m, NUM_CLASS)
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

'''====================================== [ visualization ] ======================================='''
save_root = 'visualization2/%s'%area
os.makedirs(save_root,exist_ok=True)

for i,ply_room in enumerate(ply_rooms):
    if ply_room.split('/')[-1] != 'hallway_1':
        continue
    Points = []
    Predict = []
    Gt = []
    frames = get_frames(ply_room)
    save_path = os.path.join(save_root,ply_room.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)
    for j,frame in enumerate(frames):
        batch = voxel_process(frame)
        points = batch['points']
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['img'] = batch['img'].cuda()
            batch['depth_p'] = batch['depth_p'].cuda()
        predictions,_ = unet(batch['x'],batch['img'],batch['depth_p'],batch['valid_uv'],batch['sample_ind'])
        predictions = predictions.data.cpu().numpy().astype(np.float32)
        gt = batch['y'].astype(np.uint8)

        write_ply(os.path.join(save_path, frame.split('/')[-1].replace('process', 'gt')),np.concatenate([batch['points'][:, :3], g_class2color[gt]], axis=1))

        rdx_finetune = np.random.choice(np.arange(predictions.shape[0]), int(predictions.shape[0] * 0.8), replace=False)
        predictions2 = predictions.copy()
        predictions2[rdx_finetune,gt[rdx_finetune]]=predictions2.max()+1

        pred_lab = np.argmax(predictions, axis=1).astype(np.uint8)
        write_ply(os.path.join(save_path,frame.split('/')[-1].replace('process','predict')),np.concatenate([batch['points'][:,:3], g_class2color[pred_lab]], axis=1) )

        pred_lab2 = np.argmax(predictions2, axis=1).astype(np.uint8)
        write_ply(os.path.join(save_path, frame.split('/')[-1].replace('process', 'predict_finetune0.5')),
                  np.concatenate([batch['points'][:, :3], g_class2color[pred_lab2]], axis=1))


        rdx_sample = np.random.choice(np.arange(predictions.shape[0]), int(predictions.shape[0] * 0.1), replace=False)

        Predict.append(predictions[rdx_sample])

        Gt.append(gt[rdx_sample])

        Points.append(batch['points'][rdx_sample,:6])

        print("[%d/%d] [%d/%d]" % (i, len(ply_rooms), j, len(frames)))
    Predict = np.concatenate(Predict,axis=0)
    Gt = np.concatenate(Gt, axis=0)
    Points = np.concatenate(Points, axis=0)

    K = 20

    tree = KDTree(Points[:,:3], leaf_size=50)
    dis, ind = tree.query(Points[:,:3], k=K)
    ind = ind.reshape(-1)
    Predict = np.mean(Predict[ind,:].reshape(-1,K,NUM_CLASS),axis=1)
    Predict = np.argmax(Predict,axis=1).astype(np.uint8)


    Points_Predict = np.concatenate([Points[:,:3],g_class2color[Predict]],axis=1)
    Points_Gt = np.concatenate([Points[:,:3], g_class2color[Gt]], axis=1)

    write_ply(os.path.join(save_path, 'predict.ply'), Points_Predict)
    write_ply(os.path.join(save_path, 'gt.ply'), Points_Gt)
    write_ply(os.path.join(save_path, 'real.ply'), Points)
