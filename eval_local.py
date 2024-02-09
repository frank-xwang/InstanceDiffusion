import os 
import torch 
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst

from functools import partial
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from dataset.decode_item import make_a_sentence

from utils.checkpoint import load_model_ckpt
from utils.input import prepare_batch, prepare_instance_meta, prepare_scribble_and_instmask
from utils.model import create_grounding_tokenizer, create_clip_pretrain_model, set_alpha_scale, alpha_generator

device = "cuda"

@torch.no_grad()
def run(meta_dict_list, ckpt, args, starting_noise=None):
    # load models
    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(ckpt, args, device)

    # load grounding information tokenizer
    grounding_tokenizer_input = create_grounding_tokenizer(config, model)

    # update config from args
    config.update( vars(args) )
    config = OmegaConf.create(config)

    # create clip model and processor for text prompts
    clip_model, clip_processor = create_clip_pretrain_model()

    for test_info in tqdm(meta_dict_list):
        # prepare batch inputs
        batch = prepare_batch(test_info, config.batch_size, model=clip_model, processor=clip_processor, image_size=model.image_size, use_masked_att=args.use_masked_att, device=device)

        # text prompt
        context = text_encoder.encode(  [test_info["prompt"]]*config.batch_size  )
        # text prompt for each instance
        if args.mis > 0:
            context_instances = []
            for i in range(len(batch['instance_meta'])):
                context_inst = text_encoder.encode(  [test_info['instance_meta'][i]["prompt"]]*config.batch_size  )
                context_instances.append(context_inst)

        # negative prompt
        uc = text_encoder.encode( config.batch_size*[""] )
        if args.negative_prompt is not None:
            uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )

        # sampler
        alpha_generator_func = partial(alpha_generator, type=test_info.get("alpha_type"))
        if config.mis > 0:
            sampler = PLMSSamplerInst(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale, mis=config.mis)
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)

        # grounding input
        grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=config.use_masked_att)

        # model inputs
        input = dict(x = starting_noise, timesteps = None, context = context, grounding_input = grounding_input)

        # model inputs for each instance if MIS is applied
        if config.mis > 0:
            input_all = [input]
            for i in range(len(batch['instance_meta'])):
                grounding_input_inst = grounding_tokenizer_input.prepare(batch['instance_meta'][i], return_att_masks=config.use_masked_att)
                input_inst = dict(x = starting_noise, timesteps = None, context = context_instances[i], grounding_input = grounding_input_inst)
                input_all.append(input_inst)
        else:
            input_all = input

        # start sampling
        steps = 50
        shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
        samples_fake = sampler.sample(S=steps, shape=shape, input=input_all,  uc=uc, guidance_scale=config.guidance_scale)
        samples_fake = autoencoder.decode(samples_fake)

        # folder for saving results
        output_folder = os.path.join( args.folder,  test_info["save_folder_name"])
        os.makedirs( output_folder, exist_ok=True)

        start = 0
        image_ids = list(range(start, start+config.batch_size))
        for image_id, sample in zip(image_ids, samples_fake):
            if int(image_id) == 0:
                img_name = test_info['file_name']
            else:
                img_name = "{}_{}.{}".format(test_info['file_name'], str(int(image_id)), 'jpg')
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(  os.path.join(output_folder, img_name)   )
            print("image saved at: ", os.path.join(output_folder, img_name))

# convert boxes' coordinates to the relative values (0, 1)
def convert_coco_box(bbox, img_info):
    x0 = bbox[0]/img_info['width']
    y0 = bbox[1]/img_info['height']
    x1 = (bbox[0]+bbox[2])/img_info['width']
    y1 = (bbox[1]+bbox[3])/img_info['height']
    return [x0, y0, x1, y1]


def get_point_from_box(bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [(x0 + x1)/2.0, (y0 + y1)/2.0]

def rescale_points(point, width, height):
    return [point[0]/float(width), point[1]/float(height)]


def get_args_parser():
    parser = argparse.ArgumentParser('Eval script', add_help=True)
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="") # defalt=5
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='cartoon style, painting style, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    parser.add_argument("--ckpt_path", type=str, default="", help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--use_captions", action='store_true', help="use captions in mscoco")
    parser.add_argument("--use_masked_att", action='store_true', help="use masked attention")
    parser.add_argument("--alpha", type=float,  default=0.75, help="alpha for the percentage of timestep using grounded information")
    parser.add_argument("--add_random_colors", action='store_true', help="add random colors to instances")
    parser.add_argument("--add_random_textures", action='store_true', help="add random textures to instances")
    parser.add_argument("--add_instance_colors", action='store_true', help="add random colors to instances")
    parser.add_argument("--mis", type=float, default=0.3, help="the percentage of timesteps using MIS")
    parser.add_argument("--test_config", type=str,  default='', help="config for model evaluation")
    parser.add_argument("--test_dataset", type=str,  default='coco', help="testing datasets")
    args = parser.parse_args()
    # return parser
    return args

# 8 common colors used in the evaluation on attribute binding
color_list = ["black", "white", "red", "green", "yellow", "blue", "pink", "purple"]

# 8 common textures used in the evaluation on attribute binding
texture_list = ["rubber", "fluffy", "metallic", "wooden", "plastic", "fabric", "leather", "glass"]

def annToMask(polygon, img_info):
    rles = maskUtils.frPyObjects(polygon, img_info['height'], img_info['width'])
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

def main():
    args = get_args_parser()
    max_objs = 30
    n_scribble_points = 20 # 20
    n_polygon_points = 256 # 128

    ckpt = args.ckpt_path

    # read MSCOCO
    ann_file = 'datasets/coco/annotations/instances_val2017.json'
    if args.use_captions:
        ann_file_captions = 'datasets/coco/annotations/captions_val2017.json'
        coco_caption = COCO(ann_file_captions)
    coco=COCO(ann_file)

    # sort indices for reproducible results
    image_ids = coco.getImgIds()
    image_ids.sort()

    # split the image_ids into num_jobs
    n_imgs_per_job = len(image_ids) // args.num_jobs + 1
    start_index = args.job_index * n_imgs_per_job
    end_index = min((args.job_index + 1) * n_imgs_per_job, len(image_ids))
    print("start_index: ", start_index)
    print("end_index: ", end_index)

    # start image generation
    meta_dict_list = []
    for img_id in image_ids[start_index:end_index]:
        test_info = dict(
            prompt = None,
            phrases = None,
            locations = None,
            alpha_type = [args.alpha, 0, 1.0 - args.alpha],
            file_name = None,
            save_folder_name=f"{args.save_dir}"
        )
        # Pick one image.
        img_info = coco.loadImgs([img_id])[0]
        test_info['file_name'] = img_info['file_name']

        # Get all the annotations for the specified image.
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # add captions to the prompt
        if args.use_captions:
            ann_ids_captions = coco_caption.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns_caption = coco_caption.loadAnns(ann_ids_captions)[0]['caption']

        # get bounding box coordinates for each annotation
        bbox_list = [ann["bbox"] for ann in anns]
        # Convert COCO bounding box to PIL.Image bounding box format (upper left x, upper left y, lower right x, lower right y)
        test_info['locations'] = [convert_coco_box(bbox, img_info) for bbox in bbox_list][:max_objs]

        # prepare point, scribble, instance mask conditionings for each annotation
        polygons_list = []
        scribbles_list = []
        segs = []
        points_list = []
        for ann in anns:
            scribbles, polygons, binary_mask = prepare_scribble_and_instmask(coco, ann, img_info, n_scribble_points=n_scribble_points, n_polygon_points=n_polygon_points)
            polygons_list.append(polygons)
            segs.append(binary_mask)
            if 'point' in ann:
                points_list.append(ann['point'])
            if "scribble" in ann:
                scribbles_list.append(ann['scribble'])
            else:
                scribbles_list.append(scribbles)

        # select max_objs objects          
        test_info['segs'] = np.stack(segs).astype(np.float32).squeeze()[:max_objs] if len(segs) > 0 else segs
        test_info['polygons'] = polygons_list[:max_objs]
        test_info['scribbles'] = scribbles_list[:max_objs]

        # get points for each instance, if not provided, use the center of the box
        if len(points_list) == 0:
            test_info['points'] = [get_point_from_box(box) for box in test_info['locations']][:max_objs]
        else: 
            test_info['points'] = [rescale_points(point, img_info['width'], img_info['height']) for point in points_list][:max_objs]
            
        cat_ids = [ann['category_id'] for ann in anns]

        # add random color to instance prompt if specified
        cat_inst_ids = [ann['id'] for ann in anns]
        if args.add_random_colors:
            colors = [color_list[cat_inst_id % len(color_list)] for cat_inst_id in cat_inst_ids]
        # add random texture to instance prompt if specified
        if args.add_random_textures:
            textures = [texture_list[cat_inst_id % len(texture_list)] for cat_inst_id in cat_inst_ids]

        # get categories
        cats = coco.loadCats(cat_ids)
        cat_names = [cat["name"] for cat in cats]
        if args.add_random_colors:
            cat_names = [color + " " + name for name, color in zip(cat_names, colors)]
        if args.add_random_textures:
            cat_names = [texture + " " + name for name, texture in zip(cat_names, textures)]

        test_info['phrases'] = cat_names[:max_objs]
        caption = make_a_sentence(cat_names)
        if args.use_captions:
            caption = anns_caption + caption
        test_info['prompt'] = caption
        if args.mis > 0:
            test_info['instance_meta'] = []
            for i in range(len(test_info['phrases'])):
                test_info['instance_meta'].append(prepare_instance_meta(test_info, i))
        meta_dict_list.append(test_info)

    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
    run(meta_dict_list, ckpt, args, starting_noise)

if __name__ == "__main__":
    main()