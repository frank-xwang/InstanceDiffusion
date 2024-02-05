import torch
import numpy as np
from PIL import Image
from .model import get_clip_feature
from pycocotools import mask as maskUtils
from dataset.jsondataset import batch_to_device
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask

def create_zero_input_tensors(max_objs, n_polygon_points, n_scribble_points):
    masks = torch.zeros(max_objs) # binay, indicates the instance conditioning exists or not
    text_masks = torch.zeros(max_objs) # binay, indicates the instance conditioning exists or not
    text_embeddings = torch.zeros(max_objs, 768)
    boxes_embeddings = torch.zeros(max_objs, 4)
    polygons_embeddings = torch.zeros(max_objs, n_polygon_points*2 )
    scribbles_embeddings = torch.zeros(max_objs, n_scribble_points*2 )
    segs_embeddings = torch.zeros(max_objs, 512, 512)
    points_embeddings = torch.zeros(max_objs, 2)

    return boxes_embeddings, masks, text_masks, text_embeddings, polygons_embeddings, scribbles_embeddings, segs_embeddings, points_embeddings

def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask

# modify attention mask for object idx based on the bounding box
def get_attmask_w_box(att_masks, idx, box, image_size):
    x1, y1, x2, y2 = int(np.round(box[0]*image_size)), int(np.round(box[1]*image_size)), int(np.round(box[2]*image_size)), int(np.round(box[3]*image_size))
    att_masks[idx][x1:x2, y1:y2] = 1
    return att_masks

# prepare batch for model inference given the meta data
@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30, model=None, processor=None, image_size=64, use_masked_att=False, device="cuda"):
    n_scribble_points = 20
    n_polygon_points = 256

    phrases = meta.get("phrases")
    polygons = meta.get("polygons")
    scribbles = meta.get("scribbles")
    segs = meta.get("segs")
    points = meta.get("points")

    phrases = [None]*len(phrases) if phrases==None else phrases 

    boxes, masks, text_masks, text_embeddings, polygons_embeddings, scribbles_embeddings, segs_embeddings, points_embeddings = create_zero_input_tensors(max_objs, n_polygon_points, n_scribble_points)

    if use_masked_att:
        att_masks = torch.zeros(max_objs, image_size, image_size)

    text_features = []
    for phrase in phrases:
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )

    for idx, (box, text_feature, polygon, scribble, seg, point) in enumerate(zip( meta['locations'], text_features, polygons, scribbles, segs, points)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1
        if polygon is not None:
            polygons_embeddings[idx] = torch.tensor(polygon)
        if scribble is not None:
            scribbles_embeddings[idx] = torch.tensor(scribble)
        if seg is not None:
            segs_embeddings[idx] = torch.tensor(seg)
        if point is not None:
            points_embeddings[idx] = torch.tensor(point)
    
        # get attention masks based on the bounding boxes
        if use_masked_att: att_masks = get_attmask_w_box(att_masks, idx, box, image_size)

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        'polygons': polygons_embeddings.unsqueeze(0).repeat(batch,1,1),
        'scribbles': scribbles_embeddings.unsqueeze(0).repeat(batch,1,1),
        'segs': segs_embeddings.unsqueeze(0).repeat(batch,1,1,1),
        'points': points_embeddings.unsqueeze(0).repeat(batch,1,1),
    }

    # get model inputs for each instance if MIS is applied
    if "instance_meta" in meta:
        out["instance_meta"] = []
        for i in range(len(meta['instance_meta'])):
            boxes_, masks_, text_masks_, text_embeddings_, polygons_embeddings_, scribbles_embeddings_, segs_embeddings_, points_embeddings_ = create_zero_input_tensors(max_objs, n_polygon_points, n_scribble_points)

            boxes_[0] = torch.tensor(np.array(meta["instance_meta"][i]["locations"][0]))
            polygons_embeddings_[0] = torch.tensor(np.array(meta["instance_meta"][i]["polygons"][0]))
            scribbles_embeddings_[0] = torch.tensor(np.array(meta["instance_meta"][i]["scribbles"][0]))
            segs_embeddings_[0] = torch.tensor(np.array(meta["instance_meta"][i]["segs"][0]))
            points_embeddings_[0] = torch.tensor(np.array(meta["instance_meta"][i]["points"][0]))
            masks_[0] = 1

            if text_features[i] is not None:
                text_masks_[0] = 1
                text_embeddings_[0] = text_features[i]

            out["instance_meta"].append({
                "boxes" : boxes_.unsqueeze(0).repeat(batch,1,1),
                "masks" : masks_.unsqueeze(0).repeat(batch,1),
                "text_masks" : text_masks_.unsqueeze(0).repeat(batch,1)*complete_mask( meta['instance_meta'][i].get("text_mask"), max_objs ),
                "text_embeddings"  : text_embeddings_.unsqueeze(0).repeat(batch,1,1),
                'polygons': polygons_embeddings_.unsqueeze(0).repeat(batch,1,1),
                'scribbles': scribbles_embeddings_.unsqueeze(0).repeat(batch,1,1),
                'segs': segs_embeddings_.unsqueeze(0).repeat(batch,1,1,1),
                'points': points_embeddings_.unsqueeze(0).repeat(batch,1,1),
            })
            if use_masked_att:
                att_masks_ = torch.zeros(max_objs, image_size, image_size)
                att_masks_[0] = att_masks[i]
                out["instance_meta"][i]["att_masks"] = att_masks_.unsqueeze(0).repeat(batch,1,1,1)

    if use_masked_att:
        out["att_masks"] = att_masks.unsqueeze(0).repeat(batch,1,1,1)
    return batch_to_device(out, device) 


@torch.no_grad()
# prepare instance i's meta data for model inference
def prepare_instance_meta(test_info, i, file_name=None, save_folder_name=None, ckpt=None):
    instance_meta = {
        'ckpt': test_info.get("ckpt", None),
        'phrases': [test_info['phrases'][i]],
        'locations': [test_info['locations'][i]],
        'polygons': [test_info['polygons'][i]],
        'segs': [test_info['segs'][i]],
        'scribbles': [test_info['scribbles'][i]],
        'points': [test_info['points'][i]],
        'alpha_type': test_info['alpha_type'],
        'prompt': test_info['phrases'][i], # test_info['prompt'], # 
        'file_name': file_name,
        'save_folder_name': save_folder_name,
    }
    return instance_meta

def annToMask(polygon, img_info):
    rles = maskUtils.frPyObjects(polygon, img_info['height'], img_info['width'])
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

def convert_points(points, img_info):
    # convert polygons/scribbless' coordinates to the relative values (0, 1)
    for i in range(len(points)):
        if i % 2 == 0:
            points[i] = min(points[i] / img_info['width'], 1.0)
        else:
            points[i] = min(points[i] / img_info['height'], 1.0)
    return points

def prepare_scribble_and_instmask(coco, ann, img_info, n_scribble_points=20, n_polygon_points=256):
    # get binary mask for each object
    if coco is None:
        binary_mask = annToMask(ann, img_info)
    else:
        binary_mask = coco.annToMask(ann)
    binary_mask = Image.fromarray(binary_mask)
    binary_mask = binary_mask.resize((512, 512), resample=Image.Resampling.NEAREST)
    binary_mask = np.array(binary_mask).reshape(512, 512, 1)

    # sample random points from mask as scribbles
    scribbles = sample_random_points_from_mask(binary_mask, n_scribble_points)
    scribbles = convert_points(scribbles, img_info)

    # sample random points within the mask and along the boundary
    if coco is not None:
        binary_mask_ = np.expand_dims(coco.annToMask(ann), axis=2)
    else:
        binary_mask_ = binary_mask

    polygons = sample_sparse_points_from_mask(binary_mask_, k=n_polygon_points)
    if polygons is None:
        polygons = [0 for _ in range(n_polygon_points*2)]
    polygons = convert_points(polygons, img_info)

    return scribbles, polygons, binary_mask