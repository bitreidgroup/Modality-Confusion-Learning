import numpy as np
from PIL import Image
import pdb
import os
DATA_PATH= 'Your Data Path'
data_path = DATA_PATH

rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    print("id-train ", id_train)
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    print("id-val ",id_train)
# combine train and val split   
id_train.extend(id_val) 

files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
# relabel
pid_container = set()
camid_container = set()
modal_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)

    camid = int(img_path[-15:-14])
    camid_container.add(camid)
for img_path in files_rgb:
    camid = int(img_path[-15:-14])
    camid_container.add(camid)
pid2label = {pid:label for label, pid in enumerate(pid_container)}
#camid2label = {1:0 , 2:1 , 3:2 , 4:3 , 5:4 , 6:5 }
camid2label = {camid:label for label , camid in enumerate(camid_container)}
for camid in camid_container:
    if camid in [3,6]:
        modal_container.add(1)#1 means IR images
    elif camid in [1,2,4,5]:
        modal_container.add(2)#2 means RGB images
    else:
        print("Something goes wrong.")
modal_list = list(modal_container)
modal_list.sort()
modal2label = {modal:label for label, modal in enumerate(modal_list)}
fix_image_width = 144
fix_image_height = 288
def read_imgs(train_image):
    train_img = []
    train_label = []
    train_camid = []
    train_modal = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)


        #camid
        camid = int(img_path[-15:-14])
        camid = camid2label[camid]
        train_camid.append(camid)

        #modal
        # [1,0]  means IR images
        # [0,1] means RGB images
        camid_ = int(img_path[-15:-14])
        if camid_ in [3,6]:
            modal = [1,0]  
            modal = modal2label[1]
        elif camid_ in [1,2,4,5]:
            modal = [0,1]
            modal = modal2label[2]
        train_modal.append(modal)

    return np.array(train_img), np.array(train_label) ,np.array(train_camid) ,np.array(train_modal)

# rgb imges
train_img, train_label ,train_camid ,train_modal = read_imgs(files_rgb)
np.save(data_path+'train_rgb_resized_img.npy', train_img)
np.save(data_path+'train_rgb_resized_label.npy', train_label)
np.save(data_path+'train_rgb_resized_camid.npy', train_camid)

# ir imges
train_img, train_label ,train_camid ,train_modal = read_imgs(files_ir)
np.save(data_path+'train_ir_resized_img.npy', train_img)
np.save(data_path+'train_ir_resized_label.npy', train_label)
np.save(data_path+'train_ir_resized_camid.npy', train_camid)
print("pre_process_sysu done")
