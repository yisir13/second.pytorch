#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:36:21 2021

@author: yisha
"""
import copy
import pathlib
import pickle

import fire
import numpy as np
from skimage import io as imgio
from pypcd import pypcd

import concurrent.futures as futures
import os
import re
from collections import OrderedDict



from second.core import box_np_ops
from second.core.point_cloud.point_cloud_ops import bound_points_jit
from second.data_vlp16 import kitti_common as kitti
from second.utils.progress_bar import list_bar as prog_bar


#for simplicity we use same calib para
Trv2c = np.array([
[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
[0, 0, 0, 1]
])

# cal mean from train set
rect = np.array([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
])

P2 = np.array([[719.787081,         0., 608.463003,    44.9538775],
            [        0., 719.787081, 174.545111,     0.1066855],
            [        0.,         0.,         1., 3.0106472e-03],
            [0., 0., 0., 0]
])
img_shape = np.array((375,1242))

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [str(line).strip() for line in lines]

def create_ikg_info_file(data_path='./data_vlp16/object',
                           save_path=None,
                           create_trainval=False,
                           relative_path=True,
                           reduce = False):
    test_img_ids = _read_imageset_file("./data_vlp16/ImageSets/valid.txt")
    reduce_path = reduce
    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)

    ikg_info_val = get_ikg_image_info(
        data_path,
        training=False,
        label_info=True,
        reduce=reduce_path ,### frist time is False, save the original velo path, second time is True, save reduced velo path
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    if reduce:
        filename = save_path / 'ikg_infos_test_red.pkl'
    else:
        filename = save_path / 'ikg_infos_test.pkl'
    print(f"ikg info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(ikg_info_val, f,protocol=2)####available for python2  and 3


######################## from kitti_common ##########
def get_ikg_info_path(idx,
                        prefix,
                        info_type='velodyne',
                        file_tail='.pcd',
                        training=False,
                        relative_path=True,
                        exist_check=True):
    img_idx_str = idx
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)

    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(prefix /file_path))
    if relative_path:
        return str(file_path)#training/velodyne/000001
    else:
        return str(prefix / file_path)

def get_velodyne_path(idx, prefix, training=False, relative_path=True, exist_check=True):
    return get_ikg_info_path(idx, prefix, 'velodyne', '.pcd', training,
                               relative_path, exist_check)#training/velodyne/000001.pcd
 #### reduced path is used for Frustum evaluation  #####                              
def get_reduced_path(idx, prefix, training=False, relative_path=True, exist_check=True):
    return get_ikg_info_path(idx, prefix, 'velodyne_reduced', '.bin', training,
                               relative_path, exist_check)#training/velodyne_reduced/000001.bin

def get_label_path(idx, prefix, training=False, relative_path=True, exist_check=True):
    return get_ikg_info_path(idx, prefix, 'label_2', '.txt', training,
                               relative_path, exist_check)#training/label_2/000001.txt

def get_label_anno(label_path,reduce):
    annotations = {}
    annotations.update({
        'name': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    #print(label_path)
    content = []
    for line in lines:
        data = line.strip().split(' ')
        #change string to float
        try:
            data[1:] = [float(x) for x in data[1:]]
        except ValueError:
            print(label_path)
        if reduce == True:
            if  box_np_ops.remove_outside_labels(data[1:4], rect, Trv2c,P2,img_shape):
                content.append(data)
        else:
            content.append(data)

    for x in content:
        if x[0] == 'pedestrian':
            x[0] = 'Pedestrian'
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    # (x,y,z) in lidar coordinate
    
    annotations['location'] = np.array(
        [[float(info) for info in x[1:4]] for x in content]).reshape(-1,3)
    # annotations['bbox'] = np.array(
    #     [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # # dimensions will convert hwl format to standard lhw(camera) format.
    # if LACAS max_box - min_box; else box_dim:l,w,h

    dim = []
    for x in content:
        if len(x)>7:
            x[4] = x[7] - x[4]
            x[5] = x[8] - x[5]
            x[6] = x[9] - x[6]
        dim.append(x[4:7])
    annotations['dimensions'] = np.array(dim).reshape(-1,3)

       # [[(x[7]-x[4],x[8]-x[5],x[9]-x[6])] for x in content if len(x)>7 else x[4:7]]).reshape(-1,3)
    # else:
    #     annotations['dimensions'] = np.array(
    #         [[(x[4],x[5],x[6])] for x in content]).reshape(-1,3)
    annotations['rotation_y'] = np.zeros(len(content))
    # index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    # annotations['index'] = np.array(index, dtype=np.int32)
    # annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations
def get_ikg_image_info(path,
                         training=False,#影响打开training还是testing文件夹
                         label_info=True,
                         reduce=False,
                         calib=False,
                         image_ids=90,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=False):
    """

    读取ImagesSets包含的文件名对应的label_2, velodyne
    velodyne只需要path；label_2 要数据内容
    """

    root_path = pathlib.Path(path)
    #如果image_ids 是list, isinstance=True
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        '''return image_info:
            {'image_idx':idx,
             'pointcloud_num_features':4,
             'velodyne_path: .../xxx.xxx.pcd,
             'annos':{dimensions,name,locate,rotate_y}
             }'''
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None
        if reduce:
            image_info['velodyne_path'] = get_reduced_path(idx, path, training, relative_path)
        else:
            image_info['velodyne_path'] = get_velodyne_path(idx, path, training, relative_path)#training/velodyne/000001.pcd


        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path,reduce)

        if annotations is not None:
            image_info['annos'] = annotations
            #add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)      

#### Frustum points   save as  .bin file in /velodyne_reduced#######
def create_reduced_point_cloud(data_path,
                                save_path=None,
                                back=False):
    info_path = pathlib.Path(data_path) /'ikg_infos_test.pkl'
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info['velodyne_path']#training/velodyne/000001.pcd
        v_path = pathlib.Path(data_path) / v_path#./data/IKG/object/training/velodyne/000001.pcd

        pcd = pypcd.PointCloud.from_path(v_path)
        points = np.zeros((pcd.points, 4), dtype=np.float32)
        points[:, 0] = np.transpose(pcd.pc_data['x'])
        points[:, 1] = np.transpose(pcd.pc_data['y'])
        points[:, 2] = np.transpose(pcd.pc_data['z'])
        points[:, 3] = np.transpose(pcd.pc_data['intensity'])
        points[:, 3] /= 255 #kitti's intensity is normalized to 0-1,  so ikg's intensity need to be normalized
        points_v = points

        
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    img_shape)#img_shape :rgb img's shape, so that points not outside image

        v_path_bin = info['image_idx'] +'.bin'
        if save_path is None:
            save_filename = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path_bin#v_path.name
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)

        with open(save_filename, 'w') as f:
            points_v.tofile(f)

if __name__ == '__main__':
    fire.Fire()
