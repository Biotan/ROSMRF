import os,glob,cv2
import numpy as np

def read_SemanticMap():
    f = open('/media/biolab/S3DIS/S3DIS-ALL/semantic_labels.json','r')
    labels = f.readlines()
    labels = labels[0][1:-1]
    labels = labels.split(',')
    f.close()
    return labels

CLASS_LABELS = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

area = '5'
data_root = '/media/biolab/S3DIS/S3DIS-ALL/area_%s/data'%area
save_root = '/media/biolab/Elements/S3DIS/sequence/Area_%s'%area[0]
os.makedirs(save_root,exist_ok=True)

sem_map = read_SemanticMap()

rgbs = glob.glob(os.path.join(data_root,'rgb','*.png'))

scene_camera = {}

for i,f in enumerate(rgbs):
    camera_name = f.split('/')[-1].split('_')[1]
    scene_name = '_'.join([f.split('/')[-1].split('_')[2],f.split('/')[-1].split('_')[3]])
    frame_id = f.split('/')[-1].split('_')[5]

    if scene_name in scene_camera.keys():
        if camera_name not in scene_camera[scene_name]:
            scene_camera[scene_name][camera_name]='camera_'+str(len(scene_camera[scene_name]))
    else:
        scene_camera[scene_name]={camera_name:'camera_0'}
    if not os.path.exists(os.path.join(save_root,scene_name)):
        os.mkdir(os.path.join(save_root,scene_name))
        os.mkdir(os.path.join(save_root, scene_name,'rgb'))
        os.mkdir(os.path.join(save_root, scene_name, 'depth'))
        os.mkdir(os.path.join(save_root, scene_name, 'semantic'))
        os.mkdir(os.path.join(save_root, scene_name, 'pose'))

    img = cv2.imread(f)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(save_root, scene_name,'rgb',scene_camera[scene_name][camera_name]+'_'+frame_id+'.png'),img)

    depth = cv2.imread(f.replace('rgb','depth'),-1)
    depth = cv2.resize(depth, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(save_root, scene_name, 'depth', scene_camera[scene_name][camera_name]+'_'+frame_id + '.png'), depth)

    sem = cv2.imread(f.replace('rgb', 'semantic'))

    sem = sem[:,:,2]*(16**4)+sem[:,:,1]*(16**2)+sem[:,:,0]
    new_sem = np.zeros_like(sem).astype(np.uint8)
    cls = np.unique(sem)
    for c in cls:
        if c > 13*16**4 or c==0:
            label= 12
        else:
            label = CLASS_LABELS.index(sem_map[c].strip().strip('"').split('_')[0])
        new_sem[sem==c]=label
    new_sem = cv2.resize(new_sem, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(save_root, scene_name, 'semantic', scene_camera[scene_name][camera_name]+'_'+frame_id + '.png'), new_sem)

    os.system(" cp %s %s"%(f.replace('rgb', 'pose').replace('.png','.json'),os.path.join(save_root, scene_name, 'pose', scene_camera[scene_name][camera_name]+'_'+frame_id +'.json')))
    print('[%d/%d]'%(i,len(rgbs)))


