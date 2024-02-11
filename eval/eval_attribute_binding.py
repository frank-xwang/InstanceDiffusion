import os 
import PIL
import torch 
import argparse
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPModel,CLIPTokenizer,CLIPFeatureExtractor

def get_args_parser():
    parser = argparse.ArgumentParser('Eval script', add_help=False)
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    # args = parser.parse_args()
    return parser

def clip_score(text:str, image:PIL.Image, args:argparse.Namespace):
    clip_acc = 0

    with torch.no_grad():
        if use_open_clip:
            text_token = tokenizer(text).cuda()
            txt_features = model.encode_text(text_token)

            image = preprocess(image).unsqueeze(0).cuda()
            img_features = model.encode_image(image)

        else:
            inputs = tokenizer(text,return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].cuda()
            txt_features = model.get_text_features(inputs["input_ids"])

            inputs = feature_extractor(image)
            inputs['pixel_values'] = torch.tensor(inputs['pixel_values'][0][None]).cuda()
            img_features = model.get_image_features(inputs['pixel_values'])

        img_features, txt_features = [
            x / torch.linalg.norm(x, axis=-1, keepdims=True)
            for x in [img_features, txt_features]
        ]

        clip_score = (img_features * txt_features).sum(axis=-1).cpu().numpy().item()
    
    if args.test_random_colors:
        color_gt = text.split(" ")[0]
        color_idx = color_list.index(color_gt)
        gt_idx = color_idx
    if args.test_random_textures:
        texture_gt = text.split(" ")[0]
        texture_idx = texture_list.index(texture_gt)
        gt_idx = texture_idx
    # measure the accuracy of the color prediction by evaluating the similarity between the image features and all text prompts.
    with torch.no_grad():
        similarity = (img_features * label_prompts_feats).sum(axis=-1).cpu().numpy()
        pred = np.argmax(similarity)
        if pred == gt_idx:
            clip_acc = 1
    return clip_score, clip_acc


def convert_coco_box(bbox, img_info):
    x0 = bbox[0]/img_info['width']
    y0 = bbox[1]/img_info['height']
    x1 = (bbox[0]+bbox[2])/img_info['width']
    y1 = (bbox[1]+bbox[3])/img_info['height']
    return [x0, y0, x1, y1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    parser.add_argument("--folder", type=str, default="cocoval17-0202-308082-75%grounding-captions-InstSampler-Step15-Mean-colors", help="")
    parser.add_argument("--test_random_colors", action='store_true', help="add random colors to instances and evaluate color prediction accuracies or CLIP scores")
    parser.add_argument("--test_random_textures", action='store_true', help="add random textures to instances and evaluate color prediction accuracies or CLIP scores")

    args = parser.parse_args()

    # 8 common colors used in the evaluation on attribute binding
    color_list = ["black", "white", "red", "green", "yellow", "blue", "pink", "purple"]

    # texture list: 8 common textures
    texture_list = ["rubber", "fluffy", "metallic", "wooden", "plastic", "fabric", "leather", "glass"]

    # load coco dataset
    ann_file = 'datasets/coco/annotations/instances_val2017.json'
    coco=COCO(ann_file)
    image_ids = coco.getImgIds()

    # sort indices for reproducible results
    image_ids.sort()

    args.num_jobs = 1
    args.job_index = 0
    use_open_clip = True

    n_imgs_per_job = len(image_ids) // args.num_jobs + 1
    start_index = args.job_index * n_imgs_per_job
    end_index = min((args.job_index + 1) * n_imgs_per_job, len(image_ids))

    max_objs = 30
    meta_dict_list = []
    pbar = tqdm(image_ids[start_index:end_index])
    for img_id in pbar:
        test_info = dict(
            phrases = None,
            locations = None,
            file_name = None,
        )
        # Pick one image.
        img_info = coco.loadImgs([img_id])[0]
        test_info['file_name'] = img_info['file_name']

        # Get all the annotations for the specified image.
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # get bounding box coordinates for each annotation
        bbox_list = [ann["bbox"] for ann in anns]

        # Convert COCO bounding box to PIL.Image bounding box format (upper left x, upper left y, lower right x, lower right y)
        test_info['locations'] = [convert_coco_box(bbox, img_info) for bbox in bbox_list][:max_objs]

        cat_ids = [ann['category_id'] for ann in anns]

        if args.test_random_colors:
            cat_inst_ids = [ann['id'] for ann in anns]
            colors = [color_list[cat_inst_id % len(color_list)] for cat_inst_id in cat_inst_ids]
        if args.test_random_textures:
            cat_inst_ids = [ann['id'] for ann in anns]
            textures = [texture_list[cat_inst_id % len(texture_list)] for cat_inst_id in cat_inst_ids]

        # All categories.
        cats = coco.loadCats(cat_ids)
        cat_names = [cat["name"] for cat in cats]
        if args.test_random_colors:
            cat_names = [color + " " + name for name, color in zip(cat_names, colors)]
        if args.test_random_textures:
            cat_names = [texture + " " + name for name, texture in zip(cat_names, textures)]
        test_info['phrases'] = cat_names[:max_objs]

        # save the meta dict
        meta_dict_list.append(test_info)

    if not use_open_clip:
        version = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(version)
        model = CLIPModel.from_pretrained(version).cuda()
        feature_extractor = CLIPFeatureExtractor.from_pretrained(version)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b-s32b-b82k')
        model = model.cuda()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # compute clip scores
    clip_score_list = []
    clip_acc_list = []

    # label prompts
    if args.test_random_colors:
        label_prompts = ["a {} object".format(color) for color in color_list]
    if args.test_random_textures:
        label_prompts = ["a {} object".format(texture) for texture in texture_list]
    label_prompts = [tokenizer(label_prompt).cuda() for label_prompt in label_prompts]
    label_prompts_feats = [model.encode_text(label_prompt) for label_prompt in label_prompts]
    label_prompts_feats = [x / torch.linalg.norm(x, axis=-1, keepdims=True) for x in label_prompts_feats]
    label_prompts_feats = torch.stack(label_prompts_feats)

    home_dir = "generation_samples/" 
    folder = args.folder

    pbar = tqdm(meta_dict_list)
    for meta_dict in pbar:
        file_name = meta_dict['file_name']
        locations = meta_dict['locations']
        phrases = meta_dict['phrases']

        image_path = os.path.join(home_dir, folder, file_name)

        # read images
        image = Image.open(image_path).convert("RGB")

        # crop images using bounding boxes
        images_cropped = []
        for location in locations:
            x0, y0, x1, y1 = location
            image_cropped = image.crop((x0*image.width, y0*image.height, x1*image.width, y1*image.height))
            images_cropped.append(image_cropped)

        # measure clip scores for each cropped image (instance)
        clip_score_single_image = []
        clip_acc_single_image = []

        for text, image in zip(phrases, images_cropped):
            try:
                score_single_instance, acc_single_instance = clip_score(text, image, args)
                clip_score_single_image.append(score_single_instance)
                clip_acc_single_image.append(acc_single_instance)
            except:
                print("error: ", file_name)
                continue

        if len(clip_score_single_image) != 0:
            clip_score_list.append(np.mean(clip_score_single_image))
        if len(clip_acc_single_image) != 0:
            clip_acc_list.append(np.mean(clip_acc_single_image))

        pbar.set_postfix({'accuray': np.mean(clip_acc_list), 'similarity': np.mean(clip_score_list)})

    # # write results to file
    # with open("openclip_scores.txt", "a") as f:
    #     f.write("{}: accuray={}; similarity={}".format(folder, np.mean(clip_acc_list), np.mean(clip_score_list)))
    #     f.write("\n")
    #     print("{}: accuray={}; similarity={}".format(folder, np.mean(clip_acc_list), np.mean(clip_score_list)))