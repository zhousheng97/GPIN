import numpy as np
import os
import operator
from torch import float32
from tqdm import tqdm
import argparse
from iou import bb_intersection_over_union
from diou import compute_diou
from ciou import compute_ciou
from giou import compute_giou
from bbox_dist import min_distance_of_bbox
import math
import copy

def build_bbox_tensors(infos):
    # each time have one ocr_info
    # After num_bbox, everything else should be zero
    coord_tensor = np.zeros(4)
    lens = len(infos)
    ocr_bbox = np.zeros((lens, 4))
    ocr_bbox = np.float32(ocr_bbox)

    for i in range(lens):  # 每个image的ocr数量
        
        # print(infos[i])
        bbox = infos[i]["bounding_box"]
        if "bottom_right_x" in bbox:
            coord_tensor[0] = bbox["top_left_x"]
            coord_tensor[1] = bbox["top_left_y"]
            coord_tensor[2] = bbox["bottom_right_x"]
            coord_tensor[3] = bbox["bottom_right_y"]
        elif "top_left_x" in bbox:
            coord_tensor[0] = bbox["top_left_x"]
            coord_tensor[1] = bbox["top_left_y"]
            coord_tensor[2] = bbox["top_left_x"] + bbox["width"]
            coord_tensor[3] = bbox["top_left_y"] + bbox["height"]
        else:
            coord_tensor[0] = bbox["topLeftX"]
            coord_tensor[1] = bbox["topLeftY"]
            coord_tensor[2] = bbox["topLeftX"] + bbox["width"]
            coord_tensor[3] = bbox["topLeftY"] + bbox["height"]

        ocr_bbox[i] = coord_tensor
    
    return ocr_bbox

parser = argparse.ArgumentParser("Process Dataset")
parser.add_argument("--dataset", type=str, default="textvqa", help="textvqa, stvqa, ocrvqa")
parser.add_argument("--imdb_path", type=str, default="data/textvqa/imdb/imdb_val_ocr_en.npy",)
args = parser.parse_args()

edge_path = 'data/{}/sbd_textvqa_edge_feat_full'.format(args.dataset)
if not os.path.exists(edge_path):
    os.makedirs(edge_path)

imdb = np.load(args.imdb_path, allow_pickle=True).tolist()

for fea in tqdm(imdb[1:]):
    image_path = fea['image_path']
    filepath, tempfilename = os.path.split(image_path)
    filename, extension = os.path.splitext(tempfilename)
    save_path = os.path.join(edge_path, filepath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    info_path = os.path.join(save_path, filename+'_info.npy')
    if os.path.exists(info_path):
        continue
    feat_path = os.path.join(save_path, filename+'.npy')

    fea_ = {}
    obj_ocr_fea = np.zeros((100, 50, 5))
    ocr_obj_fea = np.zeros((50, 100, 5))
    ocr_ocr_fea = np.zeros((50, 50, 5))
    obj_obj_fea = np.zeros((100, 100, 5))
    
    obj_ocr_fea_mask = np.zeros((100, 50),dtype=np.float32)
    ocr_obj_fea_mask = np.zeros((50, 100),dtype=np.float32)
    ocr_ocr_fea_mask = np.zeros((50, 50), dtype=np.float32)
    obj_obj_fea_mask = np.zeros((100, 100),dtype=np.float32)

    obj_bbox = copy.deepcopy(fea['obj_normalized_boxes'])
  #  ocr_bbox = copy.deepcopy(fea['ocr_normalized_boxes'])
    ocr_bbox = build_bbox_tensors(fea['ocr_info'])


    img_h = copy.deepcopy(fea['image_height'])
    img_w = copy.deepcopy(fea['image_width'])

    obj_bbox[:, 0] = obj_bbox[:, 0] * img_w
    obj_bbox[:, 1] = obj_bbox[:, 1] * img_h
    obj_bbox[:, 2] = obj_bbox[:, 2] * img_w
    obj_bbox[:, 3] = obj_bbox[:, 3] * img_h

    ocr_bbox[:, 0] = ocr_bbox[:, 0] * img_w
    ocr_bbox[:, 1] = ocr_bbox[:, 1] * img_h
    ocr_bbox[:, 2] = ocr_bbox[:, 2] * img_w
    ocr_bbox[:, 3] = ocr_bbox[:, 3] * img_h

    obj_xmin, obj_ymin, obj_xmax, obj_ymax = np.split(obj_bbox, 4, axis=1)
    obj_center_x = 0.5 * (obj_xmin + obj_xmax + 1)
    obj_center_y = 0.5 * (obj_ymin + obj_ymax + 1)
    ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax = np.split(ocr_bbox, 4, axis=1)
    ocr_center_x = 0.5 * (ocr_xmin + ocr_xmax + 1)
    ocr_center_y = 0.5 * (ocr_ymin + ocr_ymax + 1)
    image_diag = math.sqrt(img_h ** 2 + img_w ** 2)

    # obj-ocr
    for m, obj_box_m in enumerate(obj_bbox):
            rcx, rcy, rw, rh = (obj_box_m[0] + obj_box_m[2] + 1) / 2, (obj_box_m[1] + obj_box_m[3] + 1) / 2, (obj_box_m[2] - obj_box_m[0] + 1), (obj_box_m[3] - obj_box_m[1] + 1)
            for n, ocr_box_n in enumerate(ocr_bbox):
                if n >=50: break
                y_diff = ocr_center_y[n] - obj_center_y[m]
                x_diff = ocr_center_x[n] - obj_center_x[m]
       
                obj_ocr_edge_feats = np.array([
                (ocr_box_n[0] - rcx) / rw,
                (ocr_box_n[1] - rcy) / rh,
                (ocr_box_n[2] - rcx) / rw,
                (ocr_box_n[3] - rcy) / rh,
                ((ocr_box_n[2] - ocr_box_n[0] + 1) * (ocr_box_n[3] - ocr_box_n[1] + 1)) / (rw * rh)])

                obj_ocr_fea[m, n:(n + 1), :5] = obj_ocr_edge_feats

                obj_ocr_fea_mask[m,n] = 1


    
    # # ocr-obj
    for a, ocr_box_a in enumerate(ocr_bbox):
        rcx, rcy, rw, rh = (ocr_box_a[0] + ocr_box_a[2] + 1) / 2, (ocr_box_a[1] + ocr_box_a[3] + 1) / 2, (ocr_box_a[2] - ocr_box_a[0] + 1), (ocr_box_a[3] - ocr_box_a[1] + 1)
        if a >=50:break
        for b, obj_box_b in enumerate(obj_bbox):
            y_diff = obj_center_y[b] - ocr_center_y[a]
            x_diff = obj_center_x[b] - ocr_center_x[a]
    
            ocr_obj_edge_feats = np.array([
                (obj_box_b[0] - rcx) / rw,
                (obj_box_b[1] - rcy) / rh,
                (obj_box_b[2] - rcx) / rw,
                (obj_box_b[3] - rcy) / rh,
                ((obj_box_b[2] - obj_box_b[0] + 1) * (obj_box_b[3] - obj_box_b[1] + 1)) / (rw * rh)
            ])
            ocr_obj_fea[a, b:(b + 1), :5] = ocr_obj_edge_feats

            ocr_obj_fea_mask[a,b] = 1

    # ocr-ocr
    for a, ocr_box_a in enumerate(ocr_bbox):
        rcx, rcy, rw, rh = (ocr_box_a[0] + ocr_box_a[2] + 1) / 2, (ocr_box_a[1] + ocr_box_a[3] + 1) / 2, (ocr_box_a[2] - ocr_box_a[0] + 1), (ocr_box_a[3] - ocr_box_a[1] + 1)

        bbA_w = ocr_xmax[a] - ocr_xmin[a]
        bbA_h = ocr_ymax[a] - ocr_ymin[a]
  
        if a >=50:break
        for b, ocr_box_b in enumerate(ocr_bbox):
            if b>=50:break
   
            ocr_ocr_edge_feats = np.array([
                (ocr_box_b[0] - rcx) / rw,
                (ocr_box_b[1] - rcy) / rh,
                (ocr_box_b[2] - rcx) / rw,
                (ocr_box_b[3] - rcy) / rh,
                ((ocr_box_b[2] - ocr_box_b[0] + 1) * (ocr_box_b[3] - ocr_box_b[1] + 1)) / (rw * rh)
            ])
            ocr_ocr_fea[a, b:(b + 1), :5] = ocr_ocr_edge_feats

            ocr_ocr_fea_mask[a,b] = 1

  

    # obj-obj
    for a, obj_box_a in enumerate(obj_bbox):
        rcx, rcy, rw, rh = (obj_box_a[0] + obj_box_a[2] + 1) / 2, (obj_box_a[1] + obj_box_a[3] + 1) / 2, (obj_box_a[2] - obj_box_a[0] + 1), (obj_box_a[3] - obj_box_a[1] + 1)
     
        for b, obj_box_b in enumerate(obj_bbox):
            y_diff = obj_center_y[b] - obj_center_y[a]
            x_diff = obj_center_x[b] - obj_center_x[a]

            obj_obj_edge_feats = np.array([
                (obj_box_b[0] - rcx) / rw,
                (obj_box_b[1] - rcy) / rh,
                (obj_box_b[2] - rcx) / rw,
                (obj_box_b[3] - rcy) / rh,
                ((obj_box_b[2] - obj_box_b[0] + 1) * (obj_box_b[3] - obj_box_b[1] + 1)) / (rw * rh)
            ])
            obj_obj_fea[a, b:(b + 1), :5] = obj_obj_edge_feats

            obj_obj_fea_mask[a,b] = 1

    fea_['obj_ocr_edge_feat'] = np.float32(obj_ocr_fea)
    fea_['ocr_obj_edge_feat'] = np.float32(ocr_obj_fea)
    fea_['ocr_ocr_edge_feat'] = np.float32(ocr_ocr_fea)
    fea_['obj_obj_edge_feat'] = np.float32(obj_obj_fea)

    fea_['obj_ocr_edge_feat_mask'] = obj_ocr_fea_mask
    fea_['ocr_obj_edge_feat_mask'] = ocr_obj_fea_mask
    fea_['ocr_ocr_edge_feat_mask'] = ocr_ocr_fea_mask
    fea_['obj_obj_edge_feat_mask'] = obj_obj_fea_mask


    np.save(info_path, fea_, allow_pickle=True)
    np.save(feat_path, np.array([], dtype=np.float32).reshape([0, 4]), allow_pickle=True)