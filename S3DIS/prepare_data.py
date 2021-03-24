import open3d as o3d
import numpy as np
import cv2,os,glob,json,plyfile
import sys
sys.path.append('../')
from map import map2D_3D
from plyfile import PlyData,PlyElement
from sklearn.neighbors import KDTree

data_root = '/media/biolab/Elements/S3DIS/sequence'

ratio=0.25

def read_Matrix(path):
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
        K = np.array(load_dict['camera_k_matrix'])
        K = np.c_[K, np.zeros(3)]
        K = np.r_[K, np.array([[0, 0, 0, 1]])]

        M = np.array(load_dict['camera_rt_matrix'])
        M = np.r_[M, np.array([[0, 0, 0, 1]])]

    return K,M

def get_filenames():
    files = []
    for area in [1,2,3,4,5,6]:
        fs = glob.glob(os.path.join(data_root,'Area_%d'%area,'*/rgb/*.png'))
        files.extend(fs)
    return files

def read_ply(plypath):
    f = plyfile.PlyData.read(plypath)
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
    print(points.shape)
    print(np.unique(points[:,-1]))
    return points,valid_uv,sample_ind

def write_ply(save_path, points,valid_uv,sample_ind, text=False):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,6) [x,y,z,r,g,b,l]
    """
    vertex = list(map(lambda x: tuple(x), points[:,:6].tolist()))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'uint8'), ('green', 'uint8'),
                                     ('blue', 'uint8')])
    ver_el = PlyElement.describe(vertex, 'vertex')

    label_el = np.array(points[:,-1],dtype=[('label','uint8')])
    label_el = PlyElement.describe(label_el, 'semantic')

    valid_uv_el = np.array(valid_uv, dtype=[('valid_uv', 'uint8')])
    valid_uv_el = PlyElement.describe(valid_uv_el, 'valid_uv')

    sample_ind_el = np.array(sample_ind, dtype=[('sample_ind', 'int32')])
    sample_ind_el = PlyElement.describe(sample_ind_el, 'sample_ind')
    PlyData([ver_el,label_el,valid_uv_el,sample_ind_el], text=text).write(save_path)


def transpose_file(img_path):
    save_path = '/'.join(img_path.split('/')[:-2]).replace('sequence','sequence2')
    os.makedirs(save_path,exist_ok=True)

    fname = img_path.split('/')[-1].replace('.png','.ply')

    lbl_path = img_path.replace('rgb', 'semantic')
    depth_path = img_path.replace('rgb', 'depth')
    K, M = read_Matrix(img_path.replace('rgb', 'pose').replace('.png', '.json'))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    lbl = cv2.imread(lbl_path, -1)
    depth = cv2.imread(depth_path, -1)

    depth = depth.astype(np.float32)
    depth /=  512.

    points,valid_uv = map2D_3D(img, depth, lbl, ratio, K,M)

    # write_ply(os.path.join(save_path, fname), points)

    points,sample_ind = open3d_process(points)
    write_ply(os.path.join(save_path, fname.replace('.ply','_process.ply')), points,valid_uv.astype(np.uint8),sample_ind)


def open3d_process(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    #"Downsample the point cloud with a voxel of 0.003"
    downpcd = pcd.voxel_down_sample(voxel_size=0.025)

    #"Statistical oulier removal"
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
    downpcd = downpcd.select_down_sample(ind)

    tree = KDTree(points[:, :3],leaf_size=50)
    sub_cloud = np.asarray(downpcd.points)
    dis, ind = tree.query(sub_cloud,k=1)

    color = points[ind[:, 0], 3:6]
    label = points[ind[:,0],-1]

    points = np.c_[sub_cloud,color,label]

    return points,ind[:,0]

# file_names = get_filenames()
#
# p = mp.Pool(processes=mp.cpu_count())
# p.map(transpose_file,file_names)
# p.close()
# p.join()

# transpose_file(file_names[0])

# read_ply('/media/biolab/Elements/S3DIS/sequence2/Area_1/office_2/camera_0_0_process.ply')


def write_ply2(save_path, points, text=False):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,6) [x,y,z,r,g,b,l]
    """
    vertex = list(map(lambda x: tuple(x), points[:,:6].tolist()))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'uint8'), ('green', 'uint8'),
                                     ('blue', 'uint8')])
    ver_el = PlyElement.describe(vertex, 'vertex')

    label_el = np.array(points[:,-1],dtype=[('label','uint8')])
    label_el = PlyElement.describe(label_el, 'semantic')
    PlyData([ver_el,label_el], text=text).write(save_path)

data_root = '/media/biolab/Elements/S3DIS/sequence2/Area_5/conferenceRoom_1'
ply_files = glob.glob("{}/*.ply".format(data_root))
POINTS = []
for file in ply_files:
    points,_,_ = read_ply(file)
    POINTS.append(points)
POINTS = np.concatenate(POINTS,axis=0)
xyz_min = np.amin(POINTS, axis=0)[0:3] #todo
POINTS[:, 0:3] -= xyz_min
write_ply2(os.path.join('/'.join(data_root.split('/')[:-1]),'conferenceRoom_1.ply'),POINTS)

data2 = np.load('/home/biolab/research/ASIS/ASIS_BASE_ED2 /data/stanford_indoor3d_ins.sem/Area_5_conferenceRoom_1.npy')
write_ply2('/home/biolab/Desktop/Area_5_conferenceRoom_1.ply',data2)


