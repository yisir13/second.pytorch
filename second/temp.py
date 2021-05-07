# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
from second.utils.progress_bar import list_bar as prog_bar
import pathlib
trainroot = '/home/yisha/SparseConvNet/second.pytorch/second/NuScenes/nu_infos_val.pkl'
validroot = '/home/yisha/SparseConvNet/second.pytorch/second/NuScenes/nu_infos_val.pkl'

dbroot = '/home/yisha/SparseConvNet/second.pytorch/second/NuScenes/nu_dbinfos_valid.pkl'
root = '/home/yisha/SparseConvNet/second.pytorch/second/data'
v_path = '/second.pytorch/Test_nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801006946864.pcd.bin'



with open(trainroot,'rb') as f:
    infos = pickle.load(f)
# ped_infos = infos['Pedestrian']
# num_valid_ped = 0
# num_valid_car = 0
# num_valid_cyc = 0
# for info in ped_infos:
#     if info['num_points_in_gt']>5:
#         num_valid_ped +=1
# car_infos = infos['car']
# for info in car_infos:
#     if info['num_points_in_gt']>5:
#         num_valid_car +=1
# cyc_infos = infos['Cyclist']
# for info in cyc_infos:
#     if info['num_points_in_gt']>5:
#         num_valid_cyc +=1
    
#gt_annos = {}
# gt_annos.update({
#     'dimensions': [],
#     'location': [],
#     'rotation_y': [],
#     'name': [],
#     })

# gt_annos.update({'name': [info['gt_names'] for info in infos],
#                 'dimensions':[info['gt_boxes'][:,:3] for info in infos],
#                 'location':[info['gt_boxes'][:,3:6] for info in infos],
#                 'rotation_y':[info['gt_boxes'][:,6] for info in infos],
#                 })

    
    # infos = infos['infos']
# def modify_infos(infos):
    
#     for i in range(len(infos)):
#         for j in range(len(infos[i]['gt_names'])):
#             if infos[i]['gt_names'][j]=='pedestrian':
#                 infos[i]['gt_names'][j] ='Pedestrian'
#             elif infos[i]['gt_names'][j] =='bicycle':
#                 infos[i]['gt_names'][j] ='Cyclist'
#     return infos
# #name = infos[0]['gt_names']
# with open(validroot,'rb') as v:
#     v_infos = pickle.load(v)
#     # v_infos = v_infos['infos']
# with open(dbroot,'rb') as d:
#     db_infos = pickle.load(d)
# modified_train = modify_infos(infos)
# modified_valid = modify_infos(v_infos)
# with open('/home/yisha/SparseConvNet/second.pytorch/second/data/second.pytorch/nu_infos_train.pkl', 'wb') as k:
#         pickle.dump(modified_train, k)
# with open('/home/yisha/SparseConvNet/second.pytorch/second/data/second.pytorch/nu_infos_val.pkl', 'wb') as vinfo:
#         pickle.dump(modified_valid, vinfo)

