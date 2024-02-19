import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import skimage.draw as draw
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools import _mask as coco_mask

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def decodeToBinaryMask(rle):
    mask = coco_mask.decode(rle)
    binaryMask = mask.astype('bool') 
    return binaryMask

# convert polygon to binary mask
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def calculate_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def mask_2_box_point(binary_mask):
    segmentation = np.where(binary_mask == 1)
    if len(segmentation[0]) == 0:
        return None, None
    # get the top-left and bottom-right point of the predicted mask
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    bbox = x_min, x_max, y_min, y_max
    center_point = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    return bbox, center_point

def calculate_scribble_inside_or_not(predicted_mask, scribbles):
    inside = []
    for scribble in scribbles:
        # check if the center point of the ground-truth mask is inside the predicted mask
        if predicted_mask[scribble[1], scribble[0]] == 1:
            inside.append(1)
        else:
            inside.append(0)
    return np.mean(inside)

def calculate_point_inside_or_not(predicted_mask, ground_truth_mask):
    _, gt_center_point = mask_2_box_point(ground_truth_mask)
    if gt_center_point is None:
        return None
    # check if the center point of the ground-truth mask is inside the predicted mask
    if predicted_mask[gt_center_point[1], gt_center_point[0]] == 1:
        return 1
    else:
        return 0

def match_masks(masks1, masks2, iou_threshold):
    matched_pairs = []

    # match each mask1 to a mask2 based on IoU. 
    # 1 mask1 can only match 1 mask2 and vice versa.
    matched_mask2_idx = []
    for mask1_idx, mask1 in enumerate(masks1):
        best_iou = -1
        best_matched_idx = -1

        for mask2_idx, mask2 in enumerate(masks2):
            if mask2_idx in matched_mask2_idx:
                continue
            iou = calculate_iou(mask1, mask2)
            if iou > best_iou:
                best_iou = iou
                best_matched_idx = mask2_idx

        if best_iou >= iou_threshold:
            matched_pairs.append((mask1_idx, best_matched_idx))
            # remove the matched mask2 from the list
            matched_mask2_idx.append(best_matched_idx)

    return matched_pairs

def sample_random_points_from_mask(mask, k):
    # Find the coordinates of non-zero pixels in the binary mask
    nonzero_coords = np.transpose(np.nonzero(mask))

    # Randomly sample 'k' points
    # return all zeros if there is no non-zero pixel
    if len(nonzero_coords) == 0:
        xy_points = [0 for _ in range(k*2)]
        return xy_points

    # randomly sample with replacement if there are not enough non-zero pixels    
    if len(nonzero_coords) < k and len(nonzero_coords) > 0:
        random_indices = np.random.choice(len(nonzero_coords), k, replace=True)
    # randomly sample withiout replacement if there are enough non-zero pixels
    else:
        random_indices = np.random.choice(len(nonzero_coords), k, replace=False)
    sampled_points = nonzero_coords[random_indices]

    # order the points by their distance to (0, 0)
    center = np.array([0, 0])
    # print(sampled_points[0])
    sampled_points = sorted(sampled_points, key=lambda x: np.linalg.norm(np.array(x) - center)) # np.linalg.norm

    # concatenate x and y coordinates and return them as a list
    # [x1,y1,x2,y2,...,x_k,y_k]
    xy_points = []
    for x in sampled_points:
        xy_points.append([int(x[1]), int(x[0])])
    return xy_points

def show(j, acc=0, size=5000):
    print("[{}/{}: {}]".format(j, size, acc), end='\r', file=sys.stdout, flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scribble", action='store_true', help="test PiM for scribble-based image generation, otherwise, test point-based image generation")
    parser.add_argument("--pred_json", type=str, default="runs/segment/val/predictions.json", help="path to YOLO-V8's model predictions. Please run YOLO-V8 first.")
    args = parser.parse_args()
    
    # load ground-truth annotations and get image ids
    coco_gt = COCO('datasets/coco/annotations/instances_val2017.json')
    img_ids = coco_gt.getImgIds()
    test_scribble = args.test_scribble

    # read model prediction using coco api
    # read model prediction for scribble/point based image generation
    pred_json = args.pred_json
    coco_pred = coco_gt.loadRes(pred_json)
    print(pred_json)

    # read annotations for image id
    acc_all = []
    # for loop with progress bar
    from tqdm import tqdm
    for i, img_id in enumerate(img_ids):
        # read ground-truth masks
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        gt_mask_list = [coco_gt.annToMask(gt_ann) for gt_ann in gt_anns]

        # get prediction masks
        pred_ann_ids = coco_pred.getAnnIds(imgIds=img_id)
        pred_anns = coco_pred.loadAnns(pred_ann_ids)
        pred_mask_list = [coco_pred.annToMask(pred_ann) for pred_ann in pred_anns]

        # matches each predicted mask to a ground-truth mask based on IoU, and the result is a list of matched pairs 
        # where each pair consists of the predicted mask index and the ground-truth mask index.
        acc_per_image = []
        masks1 = gt_mask_list
        masks2 = pred_mask_list
        matched_pairs = match_masks(masks1, masks2, 0.0)
        # calculate the IoU between the predicted masks and the ground-truth masks
        for pair in matched_pairs:
            # measure the IoU
            mask1 = masks1[pair[0]]
            mask2 = masks2[pair[1]]
            if test_scribble:
                n_scribble_points = 20
                scribbles_gt = sample_random_points_from_mask(mask1, n_scribble_points)
                if np.sum([np.sum(p) for p in scribbles_gt]) == 0:
                    continue
                acc_single = calculate_scribble_inside_or_not(predicted_mask=mask2, scribbles=scribbles_gt)
            else:
                acc_single = calculate_point_inside_or_not(predicted_mask=mask2, ground_truth_mask=mask1)
            if acc_single is not None:
                acc_per_image.append(acc_single)
        if len(acc_per_image) != 0:
            acc_all.append(np.mean(acc_per_image))

        acc_all_not_nan = [x for x in acc_all if str(x) != 'nan'] 
        show(i, np.mean(acc_all_not_nan))

    print("PiM: ", np.mean(acc_all_not_nan), len(acc_all_not_nan))
