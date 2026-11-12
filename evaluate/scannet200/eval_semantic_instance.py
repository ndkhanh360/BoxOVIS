# Based on https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
# and https://github.com/JonasSchult/Mask3D/blob/main/benchmark/evaluate_semantic_instance.py

# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...

# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.

# python imports
import os, sys
from copy import deepcopy
from uuid import uuid4
import pdb
import torch
from evaluate.scannet200.scannet_constants import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200, VALID_CLASS_IDS_200, CLASS_LABELS_200,VALID_CLASS_IDS_200_INST
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

# Additional imports for comprehensive results
try:
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
except ImportError as e:
    print(f"Warning: Some packages for comprehensive results not available: {e}")
    # These imports are optional - function will handle missing imports gracefully

import evaluate.scannet200.util as util
import evaluate.scannet200.util_3d as util_3d
import copy

# opt = parser.parse_args()

#    opt.output_file = os.path.join(os.getcwd(), 'semantic_instance_evaluation.txt')



# ---------- Label info ---------- #
DATASET_NAME = "scannet" # this is the default value, it is overridden if it is another dataset.
# CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
#                 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
# VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
CLASS_LABELS = CLASS_LABELS_200
VALID_CLASS_IDS = VALID_CLASS_IDS_200

ID_TO_LABEL = {}
LABEL_TO_ID = {}
PRED_ID_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    
for i in range(len(VALID_CLASS_IDS_200_INST)):
    PRED_ID_TO_ID[i] = VALID_CLASS_IDS_200_INST[i]   
# PRED_ID_TO_ID[-1] = -1

PRED_ID_TO_ID[198] = -1

# HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200 = None, None, None

# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt['overlaps'] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt['min_region_sizes'] = np.array([100])  # 100 for scannet
# distance thresholds [m]
opt['distance_threshes'] = np.array([float('inf')])
# distance confidences
opt['distance_confs'] = np.array([-float('inf')])


def evaluate_matches(matches):
    overlaps = opt['overlaps']
    min_region_sizes = [opt['min_region_sizes'][0]]
    dist_threshes = [opt['distance_threshes'][0]]
    dist_confs = [opt['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    ar = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float) #average recall
    rc = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float) # recall
    matched_predictions_category_names = {} #set([])
    gt_category_names = {} #set([])
    
    WI = dict(zip([round(ov,2) for ov in overlaps],len(overlaps)*[0.0]))
    A_OSE = dict(zip([round(ov,2) for ov in overlaps],len(overlaps)*[0.0]))
    FPo = dict(zip([round(ov,2) for ov in overlaps],len(overlaps)*[0.0]))
    FPc = dict(zip([round(ov,2) for ov in overlaps],len(overlaps)*[0.0]))
    TPc = dict(zip([round(ov,2) for ov in overlaps],len(overlaps)*[0.0]))
    
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        matched_predictions_category_names[di] = {}
        for oi, overlap_th in enumerate(overlaps):
            matched_predictions_category_names[di][oi] = {}  #set([])
            for m in matches.keys():
                gt_category_names[m] = set([])
                matched_predictions_category_names[di][oi][m] =  set([])

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'uuid' in p:
                                pred_visited[p['uuid']] = False


            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                for m in matches:
                    
                    #group all common and tail under unknown
                    
                    
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    
                        
                    
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt[
                                        'med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        gt_category_names[m].add(ID_TO_LABEL[gt['label_id']])
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                                
                            
                            # greedy assignments
                            if pred_visited[pred['uuid']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                        gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            # print("overlap", overlap)
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match:
                            hard_false_negatives += 1
                        if found_match:
                            matched_predictions_category_names[di][oi][m].add(ID_TO_LABEL[gt['label_id']])
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                            
                            
                        found_gt = False
                        for gt in pred['matched_gt']:
                            
                            overlap = float(gt['intersection']) / (
                                        gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt[
                                    'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # recall is the first point on recall curve # from softgroup
                    #https://github.com/thangvubk/SoftGroup/blob/main/softgroup/evaluation/instance_eval.py
                    rc_current = recall[0]

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    #precision_for_conv = np.copy(precision)
                    #precision_for_conv = np.append(precision_for_conv[0], precision_for_conv)
                    #precision_for_conv = np.append(precision_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                    stepWidths_ar = stepWidths
                    #stepWidths_ar = np.convolve(precision_for_conv, [-0.5, 0, 0.5], 'valid')
                    ar_current = np.dot(recall, stepWidths_ar)

                elif has_gt:
                    ap_current = 0.0
                    ar_current = 0.0
                    rc_current = 0.0
                else:
                    ap_current = float('nan')
                    ar_current = float('nan')
                    rc_current = float('nan')
                ap[di, li, oi] = ap_current
                ar[di, li, oi] = ar_current
                rc[di, li, oi] = rc_current
            try:
                WI[round(overlap_th,2)] = FPo[round(overlap_th,2)]/(TPc[round(overlap_th,2)]+FPc[round(overlap_th,2)])
            except:
                pass
    pcdc_scores = []
    d_inf = 0
    for scene_name in gt_category_names.keys():
        pcdc_scores.append(np.asarray([len(matched_predictions_category_names[d_inf][oid][scene_name])/len(gt_category_names[scene_name]) for oid in matched_predictions_category_names[d_inf].keys()]))
    pcdc_scores = np.asarray(pcdc_scores)


    return ap, ar, rc, pcdc_scores, 0.0, 0.0


def compute_averages(aps, PRETRAINED_ON_SCANNET200=True):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    
    if DATASET_NAME == 'scannet200': # compute average scores for head, common, tail categories
        head_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        common_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        tail_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        
        if not PRETRAINED_ON_SCANNET200:
            base_score = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
            novel_score = {title:[] for title in ['ap', 'ap25%', 'ap50%']}

        
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    head_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    common_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    tail_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            else:
                raise NotImplementedError(label_name)
            
            if not PRETRAINED_ON_SCANNET200:
                if label_name not in ["floor", "wall"]:
                    if (label_name in BASE_CLASSES):
                        for ap_type in ['ap', 'ap25%', 'ap50%']:
                            base_score[ap_type].append(avg_dict["classes"][label_name][ap_type])
                    else:
                        for ap_type in ['ap', 'ap25%', 'ap50%']:
                            novel_score[ap_type].append(avg_dict["classes"][label_name][ap_type])
    if DATASET_NAME=='scannet200':
        for score_type in ['ap', 'ap25%', 'ap50%']:
            avg_dict['head_'+score_type] = np.nanmean(head_scores[score_type]) #64, orig 66
            avg_dict['common_'+score_type] = np.nanmean(common_scores[score_type]) #68, orig 68
            avg_dict['tail_'+score_type] = np.nanmean(tail_scores[score_type]) #66, orig 66
        
        if not PRETRAINED_ON_SCANNET200:
            for score_type in ['ap', 'ap25%', 'ap50%']:
                avg_dict['base_'+score_type] = np.nanmean(base_score[score_type]) 
                avg_dict['novel_'+score_type] = np.nanmean(novel_score[score_type]) 

            
        
    
    return avg_dict


def compute_averages_ar(ars):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ar']     = np.nanmean(ars[ d_inf,:,:  ])
    avg_dict['all_ar'] = np.nanmean(ars[d_inf, :, oAllBut25])
    avg_dict['all_ar_50%'] = np.nanmean(ars[d_inf, :, o50])
    avg_dict['all_ar_25%'] = np.nanmean(ars[d_inf, :, o25])
    avg_dict["classes"] = {}
        
    if DATASET_NAME == 'scannet200': # compute average scores for head, common, tail categories
        head_scores = {title:[] for title in ['ar', 'ar25%', 'ar50%']}
        common_scores = {title:[] for title in ['ar', 'ar25%', 'ar50%']}
        tail_scores = {title:[] for title in ['ar', 'ar25%', 'ar50%']}
        
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ar"]       = np.average(ars[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ar"] = np.average(ars[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ar50%"] = np.average(ars[d_inf, li, o50])
        avg_dict["classes"][label_name]["ar25%"] = np.average(ars[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    head_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    common_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    tail_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            else:
                raise NotImplementedError(label_name)
            
    if DATASET_NAME=='scannet200':
        for score_type in ['ar', 'ar25%', 'ar50%']:
            avg_dict['head_'+score_type] = np.nanmean(head_scores[score_type]) #64, orig 66
            avg_dict['common_'+score_type] = np.nanmean(common_scores[score_type]) #68, orig 68
            avg_dict['tail_'+score_type] = np.nanmean(tail_scores[score_type]) #66, orig 66
    
    return avg_dict



def compute_averages_rc(rcs):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ar']     = np.nanmean(ars[ d_inf,:,:  ])
    avg_dict['all_rc'] = np.nanmean(rcs[d_inf, :, oAllBut25])
    avg_dict['all_rc_50%'] = np.nanmean(rcs[d_inf, :, o50])
    avg_dict['all_rc_25%'] = np.nanmean(rcs[d_inf, :, o25])
    avg_dict["classes"] = {}
        
    if DATASET_NAME == 'scannet200': # compute average scores for head, common, tail categories
        head_scores = {title:[] for title in ['rc', 'rc25%', 'rc50%']}
        common_scores = {title:[] for title in ['rc', 'rc25%', 'rc50%']}
        tail_scores = {title:[] for title in ['rc', 'rc25%', 'rc50%']}
        
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}

        avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
        avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    head_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    common_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    tail_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            else:
                raise NotImplementedError(label_name)
            
    if DATASET_NAME=='scannet200':
        for score_type in ['rc', 'rc25%', 'rc50%']:
            avg_dict['head_'+score_type] = np.nanmean(head_scores[score_type]) #64, orig 66
            avg_dict['common_'+score_type] = np.nanmean(common_scores[score_type]) #68, orig 68
            avg_dict['tail_'+score_type] = np.nanmean(tail_scores[score_type]) #66, orig 66
    
    return avg_dict

def compute_averages_pcdc(pcdc_scores):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}

    avg_dict['all_pcdc'] = np.nanmean(pcdc_scores[:, oAllBut25])
    avg_dict['all_pcdc_50%'] = np.nanmean(pcdc_scores[:, o50])
    avg_dict['all_pcdc_25%'] = np.nanmean(pcdc_scores[:, o25])
        
    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
    for i in range(len(pred['pred_classes'])):
        info = {}
        # info["label_id"] = pred['pred_classes'][i]
        info["label_id"] = PRED_ID_TO_ID[pred['pred_classes'][i]]
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str):
    pred_info = make_pred_info(pred)
    
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    gt2pred = deepcopy(gt_instances)
    
        
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        
        label_id = int(pred_info[uuid]['label_id'])
        conf = pred_info[uuid]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]['mask']
        #print(pred_mask.shape , gt_ids.shape)
        #print(len(pred_mask) , len(gt_ids))
        assert (len(pred_mask) == len(gt_ids))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt['min_region_sizes'][0]:
            continue  # skip if empty
            
        
        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))
        
        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            # print("intersection", intersection)
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
                
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    # matched_gt_objects = []
    # for label in gt2pred:
    #     for gt_inst in gt2pred[label]:
    #         if gt_inst['matched_pred']:  
    #             # Generate binary mask from gt_ids and instance_id
    #             gt_mask = (gt_ids == gt_inst['instance_id'])
    #             matched_gt_objects.append({
    #                 'mask': gt_mask,
    #                 'label': label,
    #                 'matched_predictions': gt_inst['matched_pred']
    #             })

    # unmatched_gt_objects = []
    # for label in gt2pred:
    #     for gt_inst in gt2pred[label]:
    #         if not gt_inst['matched_pred']:  # No matches
    #             # Generate binary mask from gt_ids and instance_id
    #             gt_mask = (gt_ids == gt_inst['instance_id'])
    #             unmatched_gt_objects.append({
    #                 'mask': gt_mask,
    #                 'label': label
    #             })

    # unmatched_predictions = []
    # for label in pred2gt:
    #     for pred_inst in pred2gt[label]:
    #         if not pred_inst['matched_gt']:  # No GT match
    #             # Get prediction mask from pred_info using uuid
    #             unmatched_predictions.append({
    #                 'mask': pred_mask,
    #                 'label': label,
    #                 'confidence': pred_inst['confidence']
    #             })

    return gt2pred, pred2gt


def print_results(avgs):
    global DATASET_NAME
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    if DATASET_NAME=='scannet200':
        print("-" * lineLen)
        for cat_type in ['head', 'common', 'tail']:
            cat_ap_avg = avgs[cat_type+'_ap']
            cat_ap_50o = avgs[cat_type+'_ap50%']
            cat_ap_25o = avgs[cat_type+'_ap25%']
            line = "{:<15}".format(cat_type) + sep + col1
            line += "{:>15.3f}".format(cat_ap_avg) + sep
            line += "{:>15.3f}".format(cat_ap_50o) + sep
            line += "{:>15.3f}".format(cat_ap_25o) + sep
            print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def print_results_pcdc(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("PCDC") + sep
    line += "{:>15}".format("PCDC_50%") + sep
    line += "{:>15}".format("PCDC_25%") + sep
    print(line)

    all_pcdc_avg = avgs["all_pcdc"]
    all_pcdc_50o = avgs["all_pcdc_50%"]
    all_pcdc_25o = avgs["all_pcdc_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_pcdc_avg) + sep
    line += "{:>15.3f}".format(all_pcdc_50o) + sep
    line += "{:>15.3f}".format(all_pcdc_25o) + sep
    print(line)
    print("")



def print_results_ap_ar_rc_pcdc(avgs, ar_avgs, rc_avgs, pcdc_avgs, WI, A_OSE, print_mode={'ap_avgs':True, 'ar_avgs':False, 'rc_avgs':True, 'pcdc_avgs':True}):
    global DATASET_NAME
    sep = ""
    col1 = ":"
    lineLen = 160

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
    if print_mode['ar_avgs']:
        line += "{:>15}".format("AR%") + sep
        line += "{:>15}".format("AR_50%") + sep
        line += "{:>15}".format("AR_25%") + sep
    if print_mode['rc_avgs']:
        line += "{:>15}".format("AR") + sep
        line += "{:>15}".format("RC_50%") + sep
        line += "{:>15}".format("RC_25%") + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15}".format("APCDC") + sep
        line += "{:>15}".format("PCDC_50%") + sep
        line += "{:>15}".format("PCDC_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        line = "{:<15}".format(label_name) + sep + col1
        if print_mode['ap_avgs']:
            ap_avg = avgs["classes"][label_name]["ap"]
            ap_50o = avgs["classes"][label_name]["ap50%"]
            ap_25o = avgs["classes"][label_name]["ap25%"]
            line += sep + "{:>15.3f}".format(ap_avg) + sep
            line += sep + "{:>15.3f}".format(ap_50o) + sep
            line += sep + "{:>15.3f}".format(ap_25o) + sep
        if print_mode['ar_avgs']:
            ar_avg = ar_avgs["classes"][label_name]["ar"]
            ar_50o = ar_avgs["classes"][label_name]["ar50%"]
            ar_25o = ar_avgs["classes"][label_name]["ar25%"]
            line += sep + "{:>15.3f}".format(ar_avg) + sep
            line += sep + "{:>15.3f}".format(ar_50o) + sep
            line += sep + "{:>15.3f}".format(ar_25o) + sep
        if print_mode['rc_avgs']:
            rc_avg = rc_avgs["classes"][label_name]["rc"]
            rc_50o = rc_avgs["classes"][label_name]["rc50%"]
            rc_25o = rc_avgs["classes"][label_name]["rc25%"]
            line += sep + "{:>15.3f}".format(rc_avg) + sep
            line += sep + "{:>15.3f}".format(rc_50o) + sep
            line += sep + "{:>15.3f}".format(rc_25o) + sep
        print(line)

    if DATASET_NAME=='scannet200':
        print("-" * lineLen)
        for cat_type in ['head', 'common', 'tail']:
            line = "{:<15}".format(cat_type) + sep + col1
            if print_mode['ap_avgs']:
                cat_ap_avg = avgs[cat_type+'_ap']
                cat_ap_50o = avgs[cat_type+'_ap50%']
                cat_ap_25o = avgs[cat_type+'_ap25%']
                line += "{:>15.3f}".format(cat_ap_avg) + sep
                line += "{:>15.3f}".format(cat_ap_50o) + sep
                line += "{:>15.3f}".format(cat_ap_25o) + sep
            if print_mode['ar_avgs']:
                cat_ar_avg = ar_avgs[cat_type+'_ar']
                cat_ar_50o = ar_avgs[cat_type+'_ar50%']
                cat_ar_25o = ar_avgs[cat_type+'_ar25%']
                line += "{:>15.3f}".format(cat_ar_avg) + sep
                line += "{:>15.3f}".format(cat_ar_50o) + sep
                line += "{:>15.3f}".format(cat_ar_25o) + sep
            if print_mode['rc_avgs']:
                cat_rc_avg = rc_avgs[cat_type+'_rc']
                cat_rc_50o = rc_avgs[cat_type+'_rc50%']
                cat_rc_25o = rc_avgs[cat_type+'_rc25%']
                line += "{:>15.3f}".format(cat_rc_avg) + sep
                line += "{:>15.3f}".format(cat_rc_50o) + sep
                line += "{:>15.3f}".format(cat_rc_25o) + sep
            print(line)
        # print(avgs.keys())
        if 'base_ap' in avgs.keys():
            for cat_type in ['base', 'novel']:
                line = "{:<15}".format(cat_type) + sep + col1
                if print_mode['ap_avgs']:
                    cat_ap_avg = avgs[cat_type+'_ap']
                    cat_ap_50o = avgs[cat_type+'_ap50%']
                    cat_ap_25o = avgs[cat_type+'_ap25%']
                    line += "{:>15.3f}".format(cat_ap_avg) + sep
                    line += "{:>15.3f}".format(cat_ap_50o) + sep
                    line += "{:>15.3f}".format(cat_ap_25o) + sep
                print(line)
            
    

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    all_ar_avg = ar_avgs["all_ar"]
    all_ar_50o = ar_avgs["all_ar_50%"]
    all_ar_25o = ar_avgs["all_ar_25%"]

    all_rc_avg = rc_avgs["all_rc"]
    all_rc_50o = rc_avgs["all_rc_50%"]
    all_rc_25o = rc_avgs["all_rc_25%"]

    all_pcdc_avg = pcdc_avgs["all_pcdc"]
    all_pcdc_50o = pcdc_avgs["all_pcdc_50%"]
    all_pcdc_25o = pcdc_avgs["all_pcdc_25%"]

    print("-" * lineLen)
    print("")
    line = ""
    line += "{:<15}".format("what") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
    if print_mode['ar_avgs']:
        line += "{:>15}".format("AR%") + sep
        line += "{:>15}".format("AR_50%") + sep
        line += "{:>15}".format("AR_25%") + sep
    if print_mode['rc_avgs']:
        line += "{:>15}".format("AR") + sep
        line += "{:>15}".format("RC_50%") + sep
        line += "{:>15}".format("RC_25%") + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15}".format("APCDC") + sep
        line += "{:>15}".format("PCDC_50%") + sep
        line += "{:>15}".format("PCDC_25%") + sep
    print(line)


    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15.3f}".format(all_ap_avg) + sep
        line += "{:>15.3f}".format(all_ap_50o) + sep
        line += "{:>15.3f}".format(all_ap_25o) + sep
    if print_mode['ar_avgs']:
        line += "{:>15.3f}".format(all_ar_avg) + sep
        line += "{:>15.3f}".format(all_ar_50o) + sep
        line += "{:>15.3f}".format(all_ar_25o) + sep
    if print_mode['rc_avgs']:
        line += "{:>15.3f}".format(all_rc_avg) + sep
        line += "{:>15.3f}".format(all_rc_50o) + sep
        line += "{:>15.3f}".format(all_rc_25o) + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15.3f}".format(all_pcdc_avg) + sep
        line += "{:>15.3f}".format(all_pcdc_50o) + sep
        line += "{:>15.3f}".format(all_pcdc_25o) + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')


def write_comprehensive_results(avgs, ar_avgs, rc_avgs, pcdc_avgs, output_file):
    """
    Write comprehensive evaluation results including all metrics, category summaries, and visualizations.
    
    Args:
        avgs: Average Precision metrics dictionary
        ar_avgs: Average Recall metrics dictionary  
        rc_avgs: Recall metrics dictionary
        pcdc_avgs: PCDC metrics dictionary
        output_file: Base output file path (will create multiple files with different extensions)
    """
    # Check if required packages are available
    try:
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
    except ImportError as e:
        print(f"Skipping comprehensive results due to missing packages: {e}")
        return
    
    # Create output directory structure
    output_path = Path(output_file)
    base_name = output_path.stem

    # Use the same directory as the main output file for comprehensive results
    results_dir = output_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving comprehensive results to: {results_dir}")
    
    # 1. Save all metrics to JSON files
    metrics_files = {
        'avgs': avgs,
        'ar_avgs': ar_avgs, 
        'rc_avgs': rc_avgs,
        'pcdc_avgs': pcdc_avgs
    }
    
    for metric_name, metric_data in metrics_files.items():
        json_file = results_dir / f"{base_name}_{metric_name}.json"
        with open(json_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_data = convert_numpy_to_python(metric_data)
            json.dump(json_data, f, indent=2)
    
    # 2. Extract HEAD/COMMON/TAIL summary
    if DATASET_NAME == 'scannet200':
        category_summary = extract_category_summary(avgs, ar_avgs, rc_avgs)
        
        # Save category summary to CSV
        summary_file = results_dir / f"{base_name}_category_summary.csv"
        category_summary.to_csv(summary_file, index=False)
        
        # 3. Create visualizations
        create_category_visualizations(category_summary, results_dir, base_name)
        
        # 4. Create per-class breakdown
        per_class_df = create_per_class_breakdown(avgs, ar_avgs, rc_avgs)
        per_class_file = results_dir / f"{base_name}_per_class_metrics.csv"
        per_class_df.to_csv(per_class_file, index=False)
        
        print(f"Comprehensive results saved to {results_dir}")
        print(f"Category summary: {summary_file}")
        print(f"Per-class metrics: {per_class_file}")
        print(f"Visualizations: {base_name}_*.png")


def convert_numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def extract_category_summary(avgs, ar_avgs, rc_avgs):
    """Extract HEAD/COMMON/TAIL category summary metrics."""
    
    categories = ['head', 'common', 'tail']
    summary_data = []
    
    for category in categories:
        # Extract metrics (using default IoU threshold)
        precision_ap = avgs.get(f'{category}_ap', 0.0)
        precision_ap25 = avgs.get(f'{category}_ap25%', 0.0) 
        precision_ap50 = avgs.get(f'{category}_ap50%', 0.0)
        
        recall_rc = rc_avgs.get(f'{category}_rc', 0.0)
        recall_rc25 = rc_avgs.get(f'{category}_rc25%', 0.0)
        recall_rc50 = rc_avgs.get(f'{category}_rc50%', 0.0)
        
        avg_recall_ar = ar_avgs.get(f'{category}_ar', 0.0) if f'{category}_ar' in ar_avgs else 0.0
        
        # Calculate redundancy rate (1 - precision)
        redundancy_rate = 1.0 - precision_ap
        redundancy_rate25 = 1.0 - precision_ap25
        redundancy_rate50 = 1.0 - precision_ap50
        
        summary_data.append({
            'category': category.upper(),
            'precision_ap': round(precision_ap, 4),
            'precision_ap25%': round(precision_ap25, 4),
            'precision_ap50%': round(precision_ap50, 4),
            'recall_rc': round(recall_rc, 4),
            'recall_rc25%': round(recall_rc25, 4), 
            'recall_rc50%': round(recall_rc50, 4),
            'avg_recall_ar': round(avg_recall_ar, 4),
            'redundancy_rate': round(redundancy_rate, 4),
            'redundancy_rate25%': round(redundancy_rate25, 4),
            'redundancy_rate50%': round(redundancy_rate50, 4)
        })
    
    return pd.DataFrame(summary_data)


def create_category_visualizations(category_summary, results_dir, base_name):
    """Create visualization plots for category metrics."""
    
    # Set up matplotlib
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300
    
    categories = category_summary['category'].tolist()
    
    # 1. Recall vs Precision comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision and Recall bars
    x = np.arange(len(categories))
    width = 0.35
    
    precision_bars = ax1.bar(x - width/2, category_summary['precision_ap'], width, 
                            label='Precision (AP)', color='#2E8B57', alpha=0.8)
    recall_bars = ax1.bar(x + width/2, category_summary['recall_rc'], width,
                         label='Recall (RC)', color='#4169E1', alpha=0.8)
    
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision vs Recall by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in precision_bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in recall_bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    # Redundancy rates
    redundancy_bars = ax2.bar(categories, category_summary['redundancy_rate'], 
                             color='#DC143C', alpha=0.8)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Redundancy Rate (1 - Precision)')
    ax2.set_title('False Positive Rate by Category')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in redundancy_bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(results_dir / f"{base_name}_category_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Multi-IoU comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    iou_thresholds = ['25%', '50%', 'avg']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, iou in enumerate(iou_thresholds):
        if iou == 'avg':
            precision_col = 'precision_ap'
            recall_col = 'recall_rc'
        else:
            precision_col = f'precision_ap{iou}'
            recall_col = f'recall_rc{iou}'
        
        precision_vals = category_summary[precision_col]
        recall_vals = category_summary[recall_col]
        
        ax.bar(x + (i-1)*width, precision_vals, width, 
               label=f'Precision @IoU {iou}', color=colors[i], alpha=0.7)
        ax.bar(x + (i-1)*width, recall_vals, width,
               label=f'Recall @IoU {iou}', color=colors[i], alpha=0.4, hatch='//')
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Score')  
    ax.set_title('Precision and Recall Across Different IoU Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(results_dir / f"{base_name}_multi_iou_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_per_class_breakdown(avgs, ar_avgs, rc_avgs):
    """Create detailed per-class metrics breakdown."""
    
    per_class_data = []
    
    for i, class_name in enumerate(CLASS_LABELS):
        # Determine category
        if class_name in HEAD_CATS_SCANNET_200:
            category = 'HEAD'
        elif class_name in COMMON_CATS_SCANNET_200:
            category = 'COMMON'  
        elif class_name in TAIL_CATS_SCANNET_200:
            category = 'TAIL'
        else:
            category = 'UNKNOWN'
        
        # Extract metrics
        class_metrics = avgs["classes"][class_name]
        rc_metrics = rc_avgs["classes"][class_name] 
        
        per_class_data.append({
            'class_name': class_name,
            'class_id': VALID_CLASS_IDS[i],
            'category': category,
            'precision_ap': round(class_metrics["ap"], 4),
            'precision_ap25%': round(class_metrics["ap25%"], 4),
            'precision_ap50%': round(class_metrics["ap50%"], 4),
            'recall_rc': round(rc_metrics["rc"], 4),
            'recall_rc25%': round(rc_metrics["rc25%"], 4),
            'recall_rc50%': round(rc_metrics["rc50%"], 4),
            'redundancy_rate': round(1.0 - class_metrics["ap"], 4)
        })
    
    return pd.DataFrame(per_class_data)


def evaluate(preds: dict, gt_path: str, output_file: str, dataset: str = "scannet", pretrained_on_scannet200=True):
    global DATASET_NAME
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt
    global HEAD_CATS_SCANNET_200
    global COMMON_CATS_SCANNET_200
    global TAIL_CATS_SCANNET_200
    global BASE_CLASSES

    if dataset == "scannet200":
        DATASET_NAME = "scannet200"
        CLASS_LABELS = (
            'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
            'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
            'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser',
            'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
            'backpack', 'toilet paper',
            'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs',
            'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
            'washing machine', 'mirror', 'copier',
            'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
            'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
            'radiator', 'recycling bin', 'container',
            'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe',
            'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
            'bathroom stall', 'shower wall',
            'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
            'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish',
            'kitchen counter', 'doorframe',
            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod',
            'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
            'toilet seat cover dispenser',
            'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
            'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
            'headphones', 'dish rack',
            'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse',
            'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
            'projector screen', 'divider',
            'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
            'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
            'cd case', 'closet rod',
            'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack',
            'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
            'luggage', 'mattress')

        VALID_CLASS_IDS = np.array((2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
                                    29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                    54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                                    99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122,
                                    125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
                                    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193,
                                    195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286,
                                    300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
                                    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167,
                                    1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182,
                                    1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191))
        
        BASE_CLASSES = ("wall", "floor", "chair", "table", "door", "couch", "cabinet", "shelf", "desk", "office chair", "bed", "sink", "picture", "window", "toilet", "bookshelf", "curtain", "armchair", "coffee table", "refrigerator", "kitchen cabinet", "counter", "dresser", "ceiling", "bathtub", "end table", "dining table", "shower curtain", "closet", "ottoman", "bench", "sofa chair", "file cabinet", "blinds", "container", "wardrobe", "seat", "column", "shower wall", "kitchen counter", "mini fridge", "shower door", "pillar", "furniture", "storage container", "closet door", "shower floor", "bathroom counter", "closet wall", "bathroom stall door", "bathroom cabinet", "folded chair", "mattress")
        
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        
        HEAD_CATS_SCANNET_200 = set(['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 
                                'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 
                                'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 
                                'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 
                                'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 
                                'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 
                                'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 
                                'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 
                                'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe', 
                                'clothes dryer', 'radiator', 'shelf'])

        COMMON_CATS_SCANNET_200 = set(["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", 
                                "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", 
                                "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", 
                                "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", 
                                "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", 
                                "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", 
                                "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", 
                                "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", 
                                "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", 
                                "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"])
                                
        TAIL_CATS_SCANNET_200 = set(["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", 
                                "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", 
                                "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", 
                                "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", 
                                "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", 
                                "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", 
                                "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", 
                                "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", 
                                "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", 
                                "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", 
                                "fire alarm", "power strip", "calendar", "poster", "luggage"])
        
    NUM_CLASSES = len(VALID_CLASS_IDS)
    
    print('evaluating', len(preds), 'scans...')
    matches = {}
    for i, (k, v) in enumerate(preds.items()):      
        gt_file = os.path.join(gt_path, k + ".txt") #"_inst.txt"
        if not os.path.isfile(gt_file):
            util.print_error('Scan {} does not match any gt file'.format(k), user_fault=True)

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(v, gt_file)

        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print('')
    ap_scores, ar_scores, rc_scores, pcdc_scores, WI, A_OSE = evaluate_matches(matches)
    avgs = compute_averages(ap_scores, pretrained_on_scannet200)
    ar_avgs = compute_averages_ar(ar_scores)
    rc_avgs = compute_averages_rc(rc_scores)
    pcdc_avgs = compute_averages_pcdc(pcdc_scores)

    # print
    #print_results(avgs)
    print_results_ap_ar_rc_pcdc(avgs, ar_avgs, rc_avgs, pcdc_avgs, WI, A_OSE)
    write_result_file(avgs, output_file)
    
    # Write comprehensive results including category analysis and visualizations
    write_comprehensive_results(avgs, ar_avgs, rc_avgs, pcdc_avgs, output_file)
    
    return avgs, ar_avgs, rc_avgs, pcdc_avgs

def main():
    print("!!! CLI is only for debugging purposes. use `evaluate()` instead.")

    # EXAMPLE USAGE:
    pred_masks = torch.load('/PATH/TO/MASKS')
    pred_scores = torch.load('/PATH/TO/SCORES')
    pred_classes = torch.load('/PATH/TO/CLASSES')
    print(pred_masks.shape, pred_scores.shape, pred_classes.shape)#pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))
    preds = {}
    preds['SCENE_NAME'] = {
        'pred_masks': pred_masks,
        'pred_scores': pred_scores,
        'pred_classes': pred_classes}


    gt_dir = '/PATH//TO/SCENE/GT/FOLDER'
    evaluate(preds, gt_dir, output_file='temp_result.txt')

if __name__ == '__main__':
    main()
