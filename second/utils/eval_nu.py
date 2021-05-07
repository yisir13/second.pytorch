import io as sysio
import time

import numba
import numpy as np

from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou




def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]



def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


####################################
def clean_data(gt_anno, dt_anno, current_class):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car', 'tractor', 'trailer']
    ignored_gt, ignored_dt = [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        # bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        # height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt

def _prepare_data(gt_annos, dt_annos, current_class):
    '''input: gt_annos[i],dt_annos[i],current_class   第i_batch的所有帧
        return: i个物体
        gt_datas_list  gt_annos[i]['bbox'] 
        dt_datas_list  dt_annos[i]['bbox'],['score']
        ignored_gts list i个0
        ignored_dets list i个0
        dontcares [np.zeros]
        total_dc_num [np.zeros]
        total_num_valid_gt i
    '''
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class)
        num_valid_gt, ignored_gt, ignored_det = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))

        #计数有效的gt
        total_num_valid_gt += num_valid_gt
        #记录gt 和 dt 的bbox
        gt_datas = len(gt_annos[i]["name"])
        dt_datas = dt_annos[i]["score"][..., np.newaxis]
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)

    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt)

#@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds

def calculate_iou_partly(gt_annos, dt_annos, num_parts=5):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    return: overlaps, parted_overlaps, total_gt_num, total_dt_num
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    #分成n/5 部分
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if gt_annos_part:
            if dt_annos_part:
                loc = np.concatenate(
                    [a["location"][:, [0, 1]] for a in gt_annos_part], 0)
                dims = np.concatenate(
                    [a["dimensions"][:, [0, 1]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate(
                    [a["location"][:, [0, 1]] for a in dt_annos_part], 0)
                dims = np.concatenate(
                    [a["dimensions"][:, [0, 1]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                #计算bev box IOU
                overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                    np.float64)

                parted_overlaps.append(overlap_part)
        else:
            parted_overlaps.append([])
        example_idx += num_part

    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
        #parted_overlaps作用未知，但是构成了overlaps
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


#@numba.jit(nopython=True)
def compute_tp_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           min_overlap,
                           thresh=0,
                           compute_fp=True):
    
    det_size = dt_datas.shape[0]
    gt_size = gt_datas
    
    dt_scores = dt_datas
    #print(det_size)


    #gt_bboxes = gt_datas[:, :4]
    #初始化,所有detect都未被验证所以false
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    #如果该dt_scores过低，算作ignored_threshold=True
    if det_size != 0:
        for i in range(det_size):
            if (dt_scores[i,0] < thresh):
                ignored_threshold[i] = True

    NO_DETECTION = -10000000
    tp, fp, fn = 0, 0, 0

    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    #将1个groundtruth与多个detection比较
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False
        for j in range(det_size):
            #不属于此classnames的detection 跳过
            if (ignored_det[j] == -1):
                continue
            #已经有匹配的det
            if (assigned_detection[j]):
                continue
            #score低于thres的det
            if (ignored_threshold[j]):
                continue
            #第j个groundtruth和第i个detect的重叠面积
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            #if 不计算fp & overlap>min_overlap
            #记下det_idx 更新valid_detection为现有的score,原来是NO_DETECTION-100000
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            # 要计算fp & overlap>min_overlap& name在currenclass
            #更新max_overlap, det_idx
            elif (compute_fp and (overlap > min_overlap)
                  and(overlap > max_overlap)and
                  ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

        # gt有这calss的物体且没被detect ：fn+1
        if (valid_detection == NO_DETECTION):
            fn += 1

        #tp 
        else:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            #标记该detection已被匹配 不能用于和其他gt再匹配了
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            #全false才能fp+1
            #条件：name在classnames(ignored_det = 0)，threshold大于阈值，但是检测不为tp，相当于valid_det_num - tp
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1

    return tp, fp, fn, thresholds[:thresh_idx]

#@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds):
    '''计算gt_num dt_num 更新pr
    '''
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]
            gt_data = gt_datas[i]
            dt_data = dt_datas[i]
            ignored_gt = ignored_gts[i]
            ignored_det = ignored_dets[i]
            # gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            # dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            # ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            # ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]

            
            tp, fp, fn,  _ = compute_tp_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn

        gt_num += gt_nums[i]
        dt_num += dt_nums[i]

def eval_class(gt_annos,
               dt_annos,
               current_class,
               min_overlap,
               num_parts=50):
    """Kitti eval. Only support 2d/bev/3d/aos eval for now.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist

        min_overlap: float, min overlap. official: 
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] 

        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """

    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    # 意思是将101分为50部分，经过一下函数得到的是：split_parts：[2,2,2,...,2,1]
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    rets = _prepare_data(gt_annos, dt_annos, current_class)
    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets
    #print(total_num_valid_gt)
    
    thresholdss =[]
    for i in range(len(gt_annos)):
        rets = compute_tp_jit(
            overlaps[i],
            gt_datas_list[i],
            dt_datas_list[i],
            ignored_gts[i],
            ignored_dets[i],
            min_overlap=min_overlap,
            thresh=0.0,
            compute_fp=False)
        tp, fp, fn, thresholds = rets
        thresholdss += thresholds.tolist()
    # thresholds是 N_SAMPLE_PTS长度的一维数组，记录分数，递减，表示阈值
    thresholdss = np.array(thresholdss)
    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
    thresholds = np.array(thresholds)
    #print(thresholds)
    #储存tp fn fp 不要similarity
    pr = np.zeros([len(thresholds), 3])
    idx = 0
    for j, num_part in enumerate(split_parts):
        gt_datas_part = np.array(gt_datas_list[idx:idx + num_part])
        dt_datas_part = np.array(dt_datas_list[idx:idx + num_part])
        ignored_dets_part = np.array(ignored_dets[idx:idx + num_part])
        ignored_gts_part = np.array(ignored_gts[idx:idx + num_part])
        #融合数据
        fused_compute_statistics(
            parted_overlaps[j],
            pr,
            total_gt_num[idx:idx + num_part],
            total_dt_num[idx:idx + num_part],
            gt_datas_part,
            dt_datas_part,
            ignored_gts_part,
            ignored_dets_part,
            min_overlap=min_overlap,
            thresholds=thresholds)
        idx += num_part
    N_SAMPLE_PTS = 41
    precision = np.zeros([N_SAMPLE_PTS])
    recall = np.zeros([N_SAMPLE_PTS])

    for i in range(len(thresholds)):
        recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
        precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
        #无记载pr[i,3]所以不计算aos
    #插值法，第i个precision的值=从第i个算起后面所有值的最大数；
    #一般使用的是插值的方法，取 11 个点 [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 的插值所得
    for i in range(len(thresholds)):
        precision[i] = np.max(precision[i:])
        recall[i] = np.max(recall[i:])
    ret_dict = {
        "recall": recall,
        "precision": precision,
    }
    return ret_dict

def get_mAP(prec):
    sums = 0
    #N_SAMPLE_PTS = 41 决定了i取11次 0，4...,40
    for i in range(0, len(prec), 4):
        sums += prec[i]
    return sums / 11 * 100

def get_ikg_eval_result(gt_annos,dt_annos,current_classes,num_parts=50):

    min_overlaps = [0.05,0.1,0.2,0.3,0.5]
    class_to_name = {
    1: 'Pedestrian',
    2: 'Cyclist',
}
    #将current_classes的类别名转换成类别数字[1,2]
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes,(list,tuple)):
        current_classes = [current_classes]
    current_classes_init = []
    for curclas in current_classes:
        #如果current_classes = ['Pedestrian','Cyclist'],转换成[1,2]
        if isinstance(curclas,str):
            current_classes_init.append(name_to_class[curclas])
        else:
            current_classes_init.append(curclas)
    current_classes = current_classes_init
    
    ##### do eval ###
    result =''
    mAPbev = []
    for c,current_class in enumerate(current_classes):
        for i,min_overlap in enumerate(min_overlaps):
            ret_dict = eval_class(gt_annos,
               dt_annos,
               current_class,
               min_overlap,
               num_parts)
            mAP_bev = get_mAP(ret_dict['precision'])
            # print(class_to_name[current_class])
            # print('mAP_bev for IOU= %.2f : ' %min_overlap)
            # print(mAP_bev)
            result += print_str(
                (f"{class_to_name[current_class]} "
                 "AP@{:.2f}:".format(min_overlaps[i])))
            result += print_str((f"bev  AP:{mAP_bev:.2f}"))
        mAPbev.append(mAP_bev)
    return result,mAPbev
            
        








    
