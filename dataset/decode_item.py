import os 
import torch 
import random
import base64
import numpy as np
import torchvision
from io import BytesIO
from collections import Counter
from PIL import Image, ImageDraw
from tkinter.messagebox import NO
import torchvision.transforms as transforms
from .base_dataset import recalculate_box_and_verify_if_valid, recalculate_scribbles
import math
import pandas as pd
import cv2 
from tqdm import tqdm
from pycocotools import _mask as coco_mask
import base64
import typing as t
import zlib
import json
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask

# import nltk
# from nltk.corpus import stopwords

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')


def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

# convert binay mask to polygon format
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, tolerance)
    polygons = []
    # print(contours)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def decodeToBinaryMask(rle):
    mask = coco_mask.decode([rle])
    binaryMask = mask.astype('bool') 
    return binaryMask

def equally_spaced_sampling_with_replacement(points_list, sample_size):
    """
    Samples points from a list with approximately equal gaps between sampled points.
    Supports sampling with replacement when the sample size is larger than the list size.

    :param points_list: List of tuples representing the (x, y) coordinates.
    :param sample_size: The number of points to sample.
    :return: A list of tuples representing the sampled (x, y) coordinates.
    """
    # If sample size is less than or equal to the list size, proceed with normal equally spaced sampling
    if sample_size <= len(points_list):
        gap_size = len(points_list) // sample_size
        sampled_points = [points_list[i * gap_size] for i in range(sample_size)]
    else:
        # If sample size is larger, perform sampling with replacement
        sampled_points = []
        for i in range(sample_size):
            # Calculate the index with wrapping around the list
            index = (i * len(points_list)) // sample_size % len(points_list)
            sampled_points.append(points_list[index])
    
    return sampled_points

def reorder_scribbles(scribbles):
    ### order the points by their distance to (0, 0)
    center = np.array([0, 0])
    scribbles = sorted(scribbles, key=lambda x: np.linalg.norm(np.array(x) - center))
    scribbles = equally_spaced_sampling_with_replacement(scribbles, 20)    
    scribbles = sorted(scribbles, key=lambda x: np.linalg.norm(np.array(x) - center))
    return scribbles

def sample_random_points_from_mask(mask, k):
    mask = mask[:,:,0]
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

    ### order the points by their distance to (0, 0)
    # center = np.array([mask.shape[0] // 2, mask.shape[1] // 2])
    center = np.array([0, 0])
    sampled_points = sorted(sampled_points, key=lambda x: np.linalg.norm(np.array(x) - center)) # np.linalg.norm

    # concatenate x and y coordinates and return them as a list
    # [x1,y1,x2,y2,...,x_k,y_k]
    xy_points = []
    for x in sampled_points:
        xy_points.append(float(x[1]))
        xy_points.append(float(x[0]))
    return xy_points

# convert numpy array of bool mask to float mask
def binary_mask_to_int(binary_mask):
    return binary_mask.astype(np.int32)

# uniformly sample points from the mask
def sample_sparse_points(binary_mask, k, return_2d=False):
    # Find the coordinates of non-zero pixels in the binary mask
    nonzero_coords = np.array(np.nonzero(binary_mask))
    if len(nonzero_coords) == 0:
        xy_points = [0 for _ in range(k*2)]
        return xy_points

    # Calculate the total number of non-zero pixels
    num_nonzero_pixels = len(nonzero_coords)

    xy_points = []
    if k >= num_nonzero_pixels:
        for x in nonzero_coords:
            xy_points.append(float(x[1]))
            xy_points.append(float(x[0]))
        for _ in range(k - num_nonzero_pixels):
            xy_points.append(nonzero_coords[-1][1])
            xy_points.append(nonzero_coords[-1][0])
        return nonzero_coords

    # Calculate the number of points to sample in each dimension
    num_points_per_dim = int(np.sqrt(k))

    # Calculate the step size to ensure equal spacing
    step_size = max(1, num_nonzero_pixels // (num_points_per_dim ** 2))

    # Sample points with equal spacing
    sampled_points = nonzero_coords[::step_size][:k]
    if return_2d:
        sampled_points = [(x[1], x[0]) for x in sampled_points]
    else:
        for x in sampled_points:
            xy_points.append(float(x[1]))
            xy_points.append(float(x[0]))
        return xy_points


def sample_uniform_sparse_points(binary_mask, k):
    # binary_mask = binary_mask[:,:,0]
    ### Step 1: Get the indices of '1' values in the binary mask
    foreground_indices = np.argwhere(binary_mask == 1)

    if len(foreground_indices) == 0:
        return []

    selected_points = []
    if len(foreground_indices) < k:
        # randomly sample with replacement if there are not enough non-zero pixels
        for i in range(k):
            random_point = random.choice(foreground_indices)
            selected_points.append((random_point[1], random_point[0]))
    else:
        # rank the points by their distance to the mean of the foreground_indices
        center = np.mean(foreground_indices, axis=0)
        # print(center)
        foreground_indices = sorted(foreground_indices, key=lambda x: np.linalg.norm(x - center)) # np.linalg.norm
        # Calculate the number of points to select from each segment
        points_per_segment = len(foreground_indices) // k

        ####Step 2: Randomly select one point from each segment
        # print(k)
        for i in range(k):
            segment_points = foreground_indices[i * points_per_segment : (i + 1) * points_per_segment]
            # choose the middle point in each segment
            random_point = segment_points[len(segment_points) // 2]
            # random_point = random.choice(segment_points)
            selected_points.append((random_point[1], random_point[0]))

    return selected_points

def sample_sparse_points_from_mask(mask, k):
    n_points = k
    n_polygons = n_points // 2 # half points should be sampled from the polygons
    mask = mask[:,:,0]
    # sample sparse points from the polygons (boundary)
    polygons = binary_mask_to_polygon(mask, tolerance=0.0)
    # concatenate polygons to a single list
    polygons_single = []
    for polygon in polygons:
        polygons_single += polygon
    if len(polygons_single) != 0:
        # uniformly sample points from the polygon
        polygons_single = np.array(polygons_single).reshape(-1,2)
        indexes = np.linspace(0, polygons_single.shape[0] - 1, n_polygons)
        indexes = list([int(i) for i in indexes])

        polygons_single = polygons_single[indexes]
        sampled_polygons = [(x[0], x[1]) for x in polygons_single]
    else:
        return None

    # sample sparse points from the mask
    n_inside_points = n_points - len(sampled_polygons)
    inside_points = sample_uniform_sparse_points(mask, n_inside_points)

    # combine inside_points and sampled_polygons
    xy_points = inside_points + sampled_polygons

    # order the points by their distance to (0, 0)
    center = np.array([0, 0])
    xy_points = sorted(xy_points, key=lambda x: np.linalg.norm(np.array(x) - center)) # np.linalg.norm

    # return the sampled points
    sampled_points = []
    for x in xy_points:
        sampled_points.append(x[0])
        sampled_points.append(x[1])
    return sampled_points

def get_polygons_from_mask(mask, tolerance=0, n_polygon_points=256):
    mask = binary_mask_to_int(mask)
    return_polygons = True
    if return_polygons:
        # convert float mask to polygons
        polygons = binary_mask_to_polygon(mask[:,:,0], tolerance=tolerance)

        # return all zeros if there is no polygon
        if len(polygons) == 0:
            polygons = [0 for _ in range(n_polygon_points*2)]
            return polygons
        
        # concatenate polygons to a single list
        polygon = []
        for p in polygons:
            polygon += p

        # uniformly sample points the polygon
        polygon = np.array(polygon).reshape(-1,2)
        indexes = np.linspace(0, polygon.shape[0] - 1, n_polygon_points)
        indexes = [int(i) for i in indexes]
        polygon = polygon[indexes].reshape(-1)

        return polygon
    else:
        sampled_points = sample_sparse_points(mask, n_polygon_points)
        return sampled_points

def decode_item(item):
    # convert string to dict
    if "image" in item and isinstance(item['image'], Image.Image):
        return item

    item['image'] = decode_base64_to_pillow(item['image'])
    segs = []
    for anno in item['annos']:
        # anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        # anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        # anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
        if "blip_clip_embeddings" in anno:
            anno['blip_clip_embeddings'] = decode_tensor_from_string(anno['blip_clip_embeddings'])
        if 'mask' in anno:
            # sample k random points from the mask
            n_scribble_points = 20
            rle = anno['mask']
            binary_mask = decodeToBinaryMask(rle)
            segs.append(binary_mask)
            if "scribbles" in anno:
                anno['scribbles'] = anno["scribbles"]
            else:
                anno['scribbles'] = sample_random_points_from_mask(binary_mask, n_scribble_points)
            # convert mask to polygon
            n_polygon_points = 256
            polygons = sample_sparse_points_from_mask(binary_mask, k=n_polygon_points)
            if polygons != None:
                anno['polygons'] = polygons
            else:
                anno['polygons'] = [0 for _ in range(n_polygon_points*2)]
    if len(segs) > 0:
        item['segs'] = np.stack(segs).astype(np.float32).squeeze()
    return item


def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field


def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)  
        data_info.pop("data_source", None) 
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None)
        anno_info.pop("category_id", None)
        anno_info.pop("area", None)
        anno_info["data_id"] = anno_info.pop("image_id")

def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


def make_a_sentence_count_nums(obj_names):
    # count the number of duplicated strings in the list
    # ["dog", "dog", "cat"]
    obj_names = dict(Counter(obj_names))
    # {'dog': 2, 'cat': 1}
    caption = ""
    for item in obj_names:
        caption += str(obj_names[item]) + " " + item + ", "
    return caption[:-2]


def make_a_sentence(obj_names, clean=False):

    if clean:
        obj_names = [ name[:-6] if ("-other" in name) else name for name in obj_names]

    caption = ""
    tokens_positive = []
    for obj_name in obj_names:
        start_len = len(caption)
        caption += obj_name
        end_len = len(caption)
        caption += ", "
        tokens_positive.append(
            [[start_len, end_len]] # in real caption, positive tokens can be disjoint, thus using list of list
        )
    caption = caption[:-2] # remove last ", "

    return caption #, tokens_positive


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 
    
    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)


class decode:
    def __init__(self, which_layer_text='before', 
                which_layer_image="after_reproject",
                prob_use_caption=1,
                random_drop_embedding='none',
                image_size=512, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                count_dups_make_a_sentence=False,
                random_blip=0.0,
                return_att_masks=False,
                add_inst_cap_2_global=False,
                ):
        self.which_layer_text  = which_layer_text
        self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        self.image_size = image_size
        self.random_flip = random_flip
        self.random_crop = random_crop

        self.count_dups_make_a_sentence = count_dups_make_a_sentence
        self.random_blip = random_blip
        self.return_att_masks = return_att_masks

        self.add_inst_cap_2_global = add_inst_cap_2_global

        assert which_layer_text in ['before','after']
        assert which_layer_image in ['after', 'after_renorm', 'after_reproject']
        assert random_drop_embedding in ['none', 'both', 'image']
        # Last linear layer used in CLIP text encoder. Here we use it to map CLIP image embedding into penultimate text space. See Appendix in paper. 
        self.projection_matrix = torch.load('projection_matrix')

        # preprocessed CLIP feature embedding length: 768
        self.embedding_len = 768

    def mapping(self, image_embedding):
        if self.which_layer_image == 'after':
            # use CLIP image feaure, the aligned feature space with norm=1. 
            return image_embedding
        elif self.which_layer_image == 'after_renorm':
            # same as before but normalize it to 28.7, which is empirically same as text penultimate feature norm.
            return image_embedding*28.7
        elif self.which_layer_image == 'after_reproject':
            # Re-project the CLIP image feature into text penultimate space using text linear matrix and norm it into 28.7
            image_embedding = project( image_embedding.unsqueeze(0), self.projection_matrix.T )
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding / image_embedding.norm() 
            image_embedding = image_embedding * 28.7 
            return image_embedding

    def vis_getitem_data(self, out=None, return_tensor=False, name="res.jpg", print_caption=True):

        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
        canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        W, H = img.size

        if print_caption:
            caption = out["caption"]
            print(caption)
            print(" ")

        boxes = []
        for box in out["boxes"]:
            x0,y0,x1,y1 = box
            boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )
        img = draw_box(img, boxes)
        
        if return_tensor:
            return  torchvision.transforms.functional.to_tensor(img)
        else:
            img.save(name)   

    def sample_dic_meta(self):
        max_boxes_per_data = 1 # self.max_boxes_per_data
        boxes_ = torch.zeros(max_boxes_per_data, 4)
        masks_ = torch.zeros(max_boxes_per_data)
        segs_ =  torch.zeros(max_boxes_per_data, self.image_size, self.image_size)
        points_ =  torch.zeros(max_boxes_per_data, 2)

        text_embeddings_ =  torch.zeros(max_boxes_per_data, self.embedding_len)
        text_masks_ = torch.zeros(max_boxes_per_data)
        image_masks_ = torch.zeros(max_boxes_per_data)
        caption_ = ""

        dict_sample = {
            "boxes" : boxes_,
            "masks" : masks_,
            "text_masks" : text_masks_,
            "image_masks" : image_masks_,
            "text_embeddings"  : text_embeddings_,
            "segs" : segs_,
            "points" : points_,
            "caption" : caption_,
        }
        if self.return_att_masks:
            dict_sample["att_masks"] = torch.zeros(max_boxes_per_data, 64, 64)

        return dict_sample

    def transform_image(self, pil_image, segs=None):
        if self.random_crop:
            assert False
            arr = random_crop_arr(pil_image, self.image_size) 
        else:
            arr, info, segs = center_crop_arr(pil_image, self.image_size, segs=segs)
		
        info["performed_flip"] = False
        if self.random_flip and random.random()<0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True
            if segs is not None:
                segs = np.flip(segs, axis=2).copy()
		
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2,0,1])
        
        if segs is not None:
            return torch.tensor(arr), info, torch.tensor(segs)
        else:
            return torch.tensor(arr), info, None

    def __call__(self, raw_item):
        raw_item = decode_item(raw_item)
        is_det = raw_item.get('is_det', False) # if it is from detection (such as o365), then we will make a pseudo caption
        out = {}

        # -------------------- id and image ------------------- # 
        out['id'] = raw_item['data_id']
        image = raw_item['image']
        # NOTE: New Seg
        segs = raw_item['segs'] if 'segs' in raw_item else None
        image_tensor, trans_info, segs_tf = self.transform_image(image, segs=segs)
        out["image"] = image_tensor

        # -------------------- grounding token ------------------- # 
        annos = raw_item['annos']
        
        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_obj_captions = []
        all_obj_scribbles = []        
        n_scribble_points = 20 # defined by n_scribble_points in decode_item()
        all_obj_polygons = []
        n_polygon_points = 256 # defined by n_polygon_points in decode_item()
        all_obj_segs = []
        all_points = []
        if is_det:
            all_category_names = []
        text_embedding_name = 'text_embedding_before'

        for ann_idx, anno in enumerate(annos):
            x, y, w, h = anno['bbox']
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_points.append( torch.tensor([(x0+x1)/2.0, (y0+y1)/2.0]) / self.image_size )
                all_masks.append(1)

                if 'scribbles' in anno:
                    scribbles_recalculated = recalculate_scribbles(anno['scribbles'], trans_info, self.image_size)
                    all_obj_scribbles.append( torch.tensor(scribbles_recalculated) / self.image_size )
                else:
                    all_obj_scribbles.append( torch.zeros(n_scribble_points*2) )

                if 'polygons' in anno:
                    polygons_recalculated = recalculate_scribbles(anno['polygons'], trans_info, self.image_size) 
                    all_obj_polygons.append( torch.tensor(polygons_recalculated) / self.image_size )
                    # NOTE: New Seg
                    all_obj_segs.append(segs_tf[ann_idx])
                else:
                    all_obj_polygons.append( torch.zeros(n_polygon_points*2) )
                    # NOTE: New Seg
                    all_obj_segs.append(torch.zeros(self.image_size, self.image_size))

                # NOTE: set save_box_scribble to True to save box and scribbles on image
                save_vis_box_scribble_polygons = False
                # draw box and scribbles on image if save_box_scribble and "res.jpg" does not exist
                if save_vis_box_scribble_polygons and 'scribbles' in anno and 'polygons' in anno:
                    img = torchvision.transforms.functional.to_pil_image( image_tensor*0.5+0.5 )
                    # canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(image_tensor) )
                    # W, H = img.size
                    img = draw_box(img, [[float(x0),float(y0),float(x1),float(y1)]])
                    for i in range(n_scribble_points):
                        x, y = int(scribbles_recalculated[i*2]), int(scribbles_recalculated[i*2+1])
                        draw = ImageDraw.Draw(img)
                        draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
                    for i in range(n_polygon_points):
                        x, y = int(polygons_recalculated[i*2]), int(polygons_recalculated[i*2+1])
                        draw = ImageDraw.Draw(img)
                        draw.ellipse((x-2, y-2, x+2, y+2), fill='blue', outline='blue')
                    img.save("box_scribble_polygons_vis.jpg")

                if 'blip_clip_embeddings' in anno and random.uniform(0, 1) < self.random_blip:
                    all_text_embeddings.append(anno["blip_clip_embeddings"])
                    # all_text_embeddings.append((anno["blip_clip_embeddings"] + anno[text_embedding_name])/2)
                else:
                    all_text_embeddings.append(anno[text_embedding_name])
                if is_det:
                    all_category_names.append(anno["category_name"])
                    if 'caption' in anno:
                        all_obj_captions.append(anno["category_name"] + ", " + anno['caption'])
                        # all_obj_captions.append(anno['caption'])
                    else:
                        all_obj_captions.append("")
                else:
                    all_obj_captions.append("")

        # Sort according to area and choose the largest N objects   
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        points = torch.zeros(self.max_boxes_per_data, 2)
        masks = torch.zeros(self.max_boxes_per_data)
        scribbles = torch.zeros(self.max_boxes_per_data, n_scribble_points*2 )
        polygons = torch.zeros(self.max_boxes_per_data, n_polygon_points*2 )
        # NOTE: New Seg
        segs = torch.zeros(self.max_boxes_per_data, self.image_size, self.image_size)

        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)

        if self.return_att_masks:
            att_masks = torch.zeros(self.max_boxes_per_data, 64, 64)

        selected_captions = [""]*self.max_boxes_per_data
        if is_det:
            category_names = []

        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            points[i] = all_points[idx]
            masks[i] = all_masks[idx]
            scribbles[i] = all_obj_scribbles[idx]
            polygons[i] = all_obj_polygons[idx]
            # NOTE: New Seg
            segs[i] = all_obj_segs[idx]

            text_embeddings[i] =  all_text_embeddings[idx]
            if is_det:
                category_names.append(all_category_names[idx])
                selected_captions[i] = all_obj_captions[idx]
            if self.return_att_masks:
                box = boxes[i]
                image_size = 64
                x1, y1, x2, y2 = int(np.round(box[0]*image_size)), int(np.round(box[1]*image_size)), int(np.round(box[2]*image_size)), int(np.round(box[3]*image_size))
                att_masks[i][x1:x2, y1:y2] = 1

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks

        out["boxes"] = boxes
        out["points"] = points
        out["masks"] = masks # indicating how many valid objects for this image-text data
        out["scribbles"] = scribbles
        out["polygons"] = polygons
        out["segs"] = segs
        out["image_masks"] = image_masks # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings
        out["obj_captions"] = selected_captions
        if self.return_att_masks:
            out["att_masks"] =  att_masks          

        # get instance-level inputs
        out["instance_meta"] = [self.sample_dic_meta() for _ in range(self.max_boxes_per_data)]

        for i, idx in enumerate(wanted_idxs):
            # one input per box
            out["instance_meta"][i] = self.sample_dic_meta()
            out["instance_meta"][i]['boxes'][0] = all_boxes[idx]
            out["instance_meta"][i]['points'][0] = all_points[idx]
            out["instance_meta"][i]['masks'][0] = all_masks[idx]
            out["instance_meta"][i]['segs'][0] = all_obj_segs[idx]
            out["instance_meta"][i]["text_masks"][0] = 1
            out["instance_meta"][i]["text_embeddings"][0] = all_text_embeddings[idx]
            out["instance_meta"][i]["image_masks"][0] = 1
            if self.return_att_masks:
                out["instance_meta"][i]["att_masks"][0] = att_masks[i]
            out["instance_meta"][i]["caption"] = all_obj_captions[idx]

        # ------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            if is_det:
                if self.count_dups_make_a_sentence:
                    out["caption"] = make_a_sentence_count_nums(category_names)
                else:
                    out["caption"] = make_a_sentence(category_names)
                if "caption" in raw_item:
                    out["caption"] = out["caption"] + ". " + raw_item["caption"]
                if self.add_inst_cap_2_global:
                    for inst_cap in out["obj_captions"]:
                        if inst_cap != "":
                            out["caption"] += ". {}".format(inst_cap)
                            # stop words with NLTK, copied from: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
                            stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]
                            # remove stop words from out["caption"]
                            out["caption"] = ' '.join([word for word in out["caption"].split() if word.lower() not in stop_words])
                    # print(out["caption"])
            else:
                out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out


def center_crop_arr(pil_image, image_size, segs=None):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
        # resize segs to 512x512 using nearest neighbor interpolation
        if segs is not None:
            segs = [Image.fromarray(seg) for seg in segs]
            segs = [seg.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX) for seg in segs]
            segs = [np.array(seg) for seg in segs]
            segs = np.stack(segs)

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    if segs is not None:
        segs = [Image.fromarray(seg) for seg in segs]
        segs = [seg.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.NEAREST) for seg in segs]
        segs = [np.array(seg) for seg in segs]
        segs = np.stack(segs)

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    info = {"performed_scale":performed_scale, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH}

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], info, segs[:, crop_y : crop_y + image_size, crop_x : crop_x + image_size] if segs is not None else None


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
