#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:06:47 2021

@author: yisha
"""

import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import fire
import numpy as np
import torch
from google.protobuf import text_format
import torchplus


import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, ikg_input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval_tutor import get_ikg_eval_result,num_tp_ped


model_dir ='/home/yisha/SparseConvNet/second.pytorch/second/models'
#point_file = '/home/yisha/second.pytorch/second/data/object/training/velodyne/000000.bin'
config_path="/home/yisha/SparseConvNet/second.pytorch/second/configs/pedestrianvlp16.config"
#ckpt_path = "/home/kuan/ys/SparseConvNet/second.pytorch/second/models/voxelnet-26925.tckpt"
ckpt_path = "/home/yisha/SparseConvNet/second.pytorch/second/models/voxelnet-46910.tckpt"
#info_path = "/home/yisha/second.pytorch/second/data/IKG/object/ikg_infos_test.pkl"
result_path = '/home/yisha/SparseConvNet/second.pytorch/second/data_vlp16/result'
# 读取config_path文件的内容存入config
config_file_bkp = "pipeline_tutor.config"
config = pipeline_pb2.TrainEvalPipelineConfig()
with open(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, config)

#shutil.copyfile(config_path, str(model_dir / config_file_bkp))
#合并一起，不重复信息,update config
# with open(config_path, "r") as f:
#         proto_str = f.read()
#         text_format.Merge(proto_str, config)
        
# with open(info_path, 'rb') as f:
#     infos = pickle.load(f)
# info = infos[0] 
# annos = info['annos']
# # we need other objects to avoid collision when sample

# loc = annos["location"]
# dims = annos["dimensions"]
# rots = annos["rotation_y"]
# gt_names = annos["name"]
# # print(gt_names, len(loc))
# gt_boxes = np.concatenate(
#     [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

#属性赋值：
#input_cfg:  eval_input_reader: {
#   record_file_path: "/SparseConvNet/second.pytorch/second/data/object/kitti_val.tfrecord"
#   class_names: ["Cyclist", "Pedestrian"]
#   batch_size: 1
#   max_num_epochs : 160
#   prefetch_size : 25
#   max_number_of_voxels: 20000
#   shuffle_points: false
#   num_workers: 3
#   anchor_area_threshold: 1
#   remove_environment: false
#   kitti_info_path: "...kitti_infos_test.pkl"
#   kitti_root_path: "/SparseConvNet/second.pytorch/second/data/object"
# } 
input_cfg = config.eval_input_reader

#model_cfg: structure and loss parameters of net
model_cfg = config.model.second
#train_cfg: optimizer parameters and iteration steps
train_cfg = config.train_config
#class_names: ["Cyclist", "Pedestrian"]
class_names = list(input_cfg.class_names)
#post_center_limit_range: [0, -50, -2.5, 80, 50, -0.5]
center_limit_range = model_cfg.post_center_limit_range
##generate voxel, initial anchors
voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
box_coder = box_coder_builder.build(model_cfg.box_coder)
target_assigner_cfg = model_cfg.target_assigner
target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                bv_range, box_coder)

# voxel_size = voxel_generator.voxel_size
# pc_range = voxel_generator.point_cloud_range
# grid_size = voxel_generator.grid_size
# feature_map_size = grid_size[:2] // 2
# feature_map_size = [*feature_map_size, 1][::-1]
# points = np.fromfile(
#         str(point_file), dtype=np.float32,
#         count=-1).reshape([-1, 4])

# voxels, coordinates, num_points = voxel_generator.generate(
#         points, 20000)

# ret = target_assigner.generate_anchors(feature_map_size)
# anchors = ret["anchors"]
# anchors = anchors.reshape([-1, 7])
# matched_thresholds = ret["matched_thresholds"]
# unmatched_thresholds = ret["unmatched_thresholds"]

# gt_classes = np.array(
#     [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
# target_dict = target_assigner.assign(anchors,gt_boxes,gt_classes=gt_classes,
#             matched_thresholds=matched_thresholds,
#             unmatched_thresholds=unmatched_thresholds)

net = second_builder.build(model_cfg, voxel_generator, target_assigner)
net.cuda()
net.eval()

torchplus.train.restore(ckpt_path, net)

eval_dataset = ikg_input_reader_builder.build(
    input_cfg,
    model_cfg,
    training=False,
    voxel_generator=voxel_generator,
    target_assigner=target_assigner
    )
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=input_cfg.batch_size,
    shuffle=False,
    num_workers=input_cfg.num_workers,
    pin_memory=False,
    collate_fn=merge_second_batch)

float_dtype = torch.float32
dt_annos = []
def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    

    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    annos = []
    result_lines = []
    for i, preds_dict in enumerate(predictions_dicts):
        img_idx = preds_dict["image_idx"]
        
        if preds_dict["box3d_lidar"] is not None:
            # box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            # box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = {}
            anno.update({
                'name': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
            })
            num_example = 0
            
            for box_lidar,score,label in zip (box_preds_lidar,scores,label_preds):

                res_line = []
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    #取box_lidar 在range范围内的检测物体
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                # bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                res_line += [str(class_names[int(label)])]
                # anno["truncated"].append(0.0)
                # anno["occluded"].append(0)
                # anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                #                       box[6])
                # anno["bbox"].append(bbox)
                
                anno["location"].append(box_lidar[:3])
                res_line += [str(l) for l in box_lidar[:3]]
                
                anno["dimensions"].append(box_lidar[3:6])
                res_line += [str(d) for d in box_lidar[3:6]]
                
                anno["rotation_y"].append(0)
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
                res_line =" ".join(res_line)
                result_lines.append(res_line)
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
                
            else:
                annos.append(empty_anno())
        else:
            annos.append(empty_anno())
            #result_lines = []
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array([img_idx] * num_example)
        result_file = f"{result_path}/{img_idx}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)
    return annos
def empty_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations
def example_convert_to_torch(example, dtype=torch.float32,
                              device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch

for example in iter(eval_dataloader):
    example = example_convert_to_torch(example, float_dtype)
    dt_annos += predict_kitti_to_anno(
            net, example, class_names, center_limit_range,
            model_cfg.lidar_input, global_set=None)
    
gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]


def gt_annos_filtered(gt_annos,max_range=2):
    """
    input gt_annos, max_range:2~20
    ouput gt_annos_f: list len = num_frames
            - dict: size = 4 
                - dimensions(4,3) location(4,3) name(4,) rotation_y(4,)
    """
    gt_annos_f=[]
    num_gt = 0
    for gt_anno in gt_annos:
        if gt_anno["location"] is not None:
            gt_loc = gt_anno["location"]
            gt_dim = gt_anno["dimensions"]
            gt_name = gt_anno["name"]
            
            #create new gt_anno_f to record filtered gt in every frame
            gt_anno_f = {}
            gt_anno_f.update({
                #'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'name': [],
            })
            
            for loc,dim,name in zip(gt_loc,gt_dim,gt_name):
                #if loc of gt in x-y plane < radius
                if np.linalg.norm(loc[:2]) < max_range:
                    gt_anno_f["name"].append(name)
                    gt_anno_f["location"].append(loc)
                    gt_anno_f["dimensions"].append(dim)
                    gt_anno_f["rotation_y"].append(0)
                    num_gt += 1
            ## 避免stack的v为空
            if gt_anno_f["name"] != []:
                try:
                    gt_anno_f = {n: np.stack(v) for n, v in gt_anno_f.items()}
                    gt_annos_f.append(gt_anno_f)
                except ValueError:
                    print(gt_anno_f)
            else:
                gt_annos_f.append(empty_anno())
        else:
            gt_annos_f.append(empty_anno())
                                            
    return gt_annos_f,num_gt

def dt_annos_filtered(dt_annos,max_range=2):
    dt_annos_f = []
    for dt_anno in dt_annos:
        if dt_anno["location"] is not None:
            dt_imageidx = dt_anno["image_idx"]
            dt_loc = dt_anno["location"]
            dt_dim = dt_anno["dimensions"]
            dt_name = dt_anno["name"]
            dt_score = dt_anno["score"]
            
            dt_anno_f = {}
            dt_anno_f.update({
                'image_idx': [],
                'dimensions': [],
                'location': [],
                'name': [],
                'rotation_y': [],
                'score': []
                })
            
            for loc,dim,name,score,idx in zip(dt_loc,dt_dim,dt_name,dt_score,dt_imageidx):
                if np.linalg.norm(loc[:2]) < max_range:
                    dt_anno_f["image_idx"].append(idx)
                    dt_anno_f["dimensions"].append(dim)
                    dt_anno_f["location"].append(loc)
                    dt_anno_f["name"].append(name)
                    dt_anno_f["rotation_y"].append(0)
                    dt_anno_f["score"].append(score)
            if dt_anno_f["name"] != []:
                dt_anno_f = {n: np.stack(v) for n,v in dt_anno_f.items()}
                dt_annos_f.append(dt_anno_f)
            else:
                dt_annos_f.append(empty_anno())
        else:
            dt_annos_f.append(empty_anno())
            
    return dt_annos_f

############### mAP ####################
get_ikg_eval_result(gt_annos,dt_annos,class_names[1])               
                    
############### FOV tue positives #########
tps_list = []
gts_list = []
dts_list = []
for r in range(1,35,1):
    gt_annos_new,num_gt = gt_annos_filtered(gt_annos,r)
    dt_annos_new = dt_annos_filtered(dt_annos,r)
    gts_list.append(num_gt)

    tps,dts = num_tp_ped(gt_annos_new,dt_annos_new,1)
    tps_list.append(tps)
    dts_list.append(dts)

pre = tps/dts
rec = tps/gts_list[-1]
