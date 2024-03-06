import os
import json
import torch 
import argparse
import numpy as np

from functools import partial
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tkinter.messagebox import NO
from diffusers.utils import load_image
from diffusers import StableDiffusionXLImg2ImgPipeline

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask, decodeToBinaryMask, reorder_scribbles

from skimage.transform import resize
from utils.checkpoint import load_model_ckpt
from utils.input import convert_points, prepare_batch, prepare_instance_meta
from utils.model import create_clip_pretrain_model, set_alpha_scale, alpha_generator

device = "cuda"

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

@torch.no_grad()
def get_model_inputs(meta, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise=None, instance_input=False):
    if not instance_input:
        # update config from args
        config.update( vars(args) )
        config = OmegaConf.create(config)

    # prepare a batch of samples
    batch = prepare_batch(meta, batch=config.num_images, max_objs=30, model=clip_model, processor=clip_processor, image_size=model.image_size, use_masked_att=True, device="cuda")
    context = text_encoder.encode(  [meta["prompt"]]*config.num_images  )

    # unconditional input
    if not instance_input:
        uc = text_encoder.encode( config.num_images*[""] )
        if args.negative_prompt is not None:
            uc = text_encoder.encode( config.num_images*[args.negative_prompt] )
    else:
        uc = None

    # sampler
    if not instance_input:
        alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
        if config.mis > 0:
            sampler = PLMSSamplerInst(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale, mis=config.mis)
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50
    else:
        sampler, steps = None, None

    # grounding input
    grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=return_att_masks)

    # model inputs
    input = dict(x = starting_noise, timesteps = None, context = context, grounding_input = grounding_input)
    return input, sampler, steps, uc, config

@torch.no_grad()
def run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise=None, guidance_scale=None):
    # prepare models inputs
    input, sampler, steps, uc, config = get_model_inputs(meta, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, instance_input=False)
    if guidance_scale is not None:
        config.guidance_scale = guidance_scale
    
    # prepare models inputs for each instance if MIS is used
    if args.mis > 0:
        input_all = [input]
        for i in range(len(meta['phrases'])):
            meta_instance = prepare_instance_meta(meta, i)
            input_instance, _, _, _, _ = get_model_inputs(meta_instance, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, instance_input=True)
            input_all.append(input_instance)
    else:
        input_all = input

    # start sampling
    shape = (config.num_images, model.in_channels, model.image_size, model.image_size)
    with torch.autocast(device_type=device, dtype=torch.float16):
        samples_fake = sampler.sample(S=steps, shape=shape, input=input_all,  uc=uc, guidance_scale=config.guidance_scale)
    samples_fake = autoencoder.decode(samples_fake)

    # define output folder
    output_folder = os.path.join( args.output,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.num_images))
    # print(image_ids)
    
    # visualize the boudning boxes
    image_boxes = draw_boxes( meta["locations"], meta["phrases"], meta["prompt"] + ";alpha=" + str(meta['alpha_type'][0]) )
    img_name = os.path.join( output_folder, str(image_ids[0])+'_boxes.png' )
    image_boxes.save( img_name )
    print("saved image with boxes at {}".format(img_name))
    
    # if use cascade model, we will use SDXL-Refiner to refine the generated images
    if config.cascade_strength > 0:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe = pipe.to("cuda:0")
        strength, steps = config.cascade_strength, 20 # default setting, need to be manually tuned.

    # save the generated images
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        if config.cascade_strength > 0:
            prompt = meta["prompt"]
            refined_image = pipe(prompt, image=sample, strength=strength, num_inference_steps=steps).images[0]
            refined_image.save(  os.path.join(output_folder, img_name.replace('.png', '_xl_s{}_n{}.png'.format(strength, steps)))   )
        sample.save(  os.path.join(output_folder, img_name)   )

def rescale_box(bbox, width, height):
    x0 = bbox[0]/width
    y0 = bbox[1]/height
    x1 = (bbox[0]+bbox[2])/width
    y1 = (bbox[1]+bbox[3])/height
    return [x0, y0, x1, y1]

def get_point_from_box(bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [(x0 + x1)/2.0, (y0 + y1)/2.0]

def rescale_points(point, width, height):
    return [point[0]/float(width), point[1]/float(height)]

def rescale_scribbles(scribbles, width, height):
    return [[scribble[0]/float(width), scribble[1]/float(height)] for scribble in scribbles]
    
# draw boxes given a lits of boxes: [[top left cornor, top right cornor, width, height],]
# show descriptions per box if descriptions is not None
def draw_boxes(boxes, descriptions=None, caption=None):
    width, height = 512, 512
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)   
    boxes = [ [ int(x*width) for x in box ] for box in boxes]
    for i, box in enumerate(boxes):
        draw.rectangle( ( (box[0], box[1]), (box[2], box[3]) ), outline=(0,0,0), width=2)
    if descriptions is not None:
        for idx, box in enumerate(boxes):
            draw.text((box[0], box[1]), descriptions[idx], fill="black")
    if caption is not None:
        draw.text((0, 0), caption, fill=(255,102,102))
    return image

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,  default="OUTPUT", help="root folder for output")
    parser.add_argument("--num_images", type=int, default=8, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--input_json", type=str, default='demos/demo_cat_dog_robin.json', help="json files for instance-level conditions")
    parser.add_argument("--ckpt", type=str, default='pretrained/instancediffusion_sd15.pth', help="pretrained checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--alpha", type=float, default=0.75, help="the percentage of timesteps using grounding inputs")
    parser.add_argument("--mis", type=float, default=0.36, help="the percentage of timesteps using MIS")
    parser.add_argument("--cascade_strength", type=float, default=0.35, help="strength of SDXL Refiner.")
    parser.add_argument("--test_config", type=str, default="configs/test_mask.yaml", help="config for model inference.")

    args = parser.parse_args()

    return_att_masks = False
    ckpt = args.ckpt

    seed = args.seed
    save_folder_name = f"gc{args.guidance_scale}-seed{seed}-alpha{args.alpha}"

    # read json files
    with open(args.input_json) as f:
        data = json.load(f)

    # START: READ BOXES AND BINARY MASKS
    boxes = []
    binay_masks = []
    # class_names = []
    instance_captions = []
    points_list = []
    scribbles_list = []
    prompt = data['caption']
    crop_mask_image = False
    for inst_idx in range(len(data['annos'])):
        if "mask" not in data['annos'][inst_idx] or data['annos'][inst_idx]['mask'] == []:
            instance_mask = np.zeros((512,512,1))
        else:
            instance_mask = decodeToBinaryMask(data['annos'][inst_idx]['mask'])
            if crop_mask_image:
                # crop the instance_mask to 512x512, centered at the center of the instance_mask image
                # get the center of the instance_mask
                center = np.array([instance_mask.shape[0]//2, instance_mask.shape[1]//2])
                # get the top left corner of the crop
                top_left = center - np.array([256, 256])
                # get the bottom right corner of the crop
                bottom_right = center + np.array([256, 256])
                # crop the instance_mask
                instance_mask = instance_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                binay_masks.append(instance_mask)
                data['width'] = 512
                data['height'] = 512
            else:
                binay_masks.append(instance_mask)

        if "bbox" not in data['annos'][inst_idx]: 
            boxes.append([0,0,0,0])
        else:
            boxes.append(data['annos'][inst_idx]['bbox'])
        if 'point' in data['annos'][inst_idx]:
            points_list.append(data['annos'][inst_idx]['point'])
        if "scribble" in data['annos'][inst_idx]:
            scribbles_list.append(data['annos'][inst_idx]['scribble'])
        # class_names.append(data['annos'][inst_idx]['category_name'])
        instance_captions.append(data['annos'][inst_idx]['caption'])
        # show_binary_mask(binay_masks[inst_idx])

    # END: READ BOXES AND BINARY MASKS
    img_info = {}
    img_info['width'] = data['width']
    img_info['height'] = data['height']

    locations = [rescale_box(box, img_info['width'], img_info['height']) for box in boxes]
    phrases = instance_captions

    # get points for each instance, if not provided, use the center of the box
    if len(points_list) == 0:
        points = [get_point_from_box(box) for box in locations] 
    else: 
        points = [rescale_points(point, img_info['width'], img_info['height']) for point in points_list] 

    # get binary masks for each instance, if not provided, use all zeros
    binay_masks = []
    for i in range(len(locations) - len(binay_masks)):
        binay_masks.append(np.zeros((512,512,1)))

    # get scribbles for each instance, if not provided, use random scribbles
    if len(scribbles_list) == 0:
        for idx, mask_fg in enumerate(binay_masks):
            # get scribbles from segmentation if scribble is not provided
            scribbles = sample_random_points_from_mask(mask_fg, 20)
            scribbles = convert_points(scribbles, img_info)
            scribbles_list.append(scribbles)
    else:
        scribbles_list = [rescale_scribbles(scribbles, img_info['width'], img_info['height']) for scribbles in scribbles_list]
        scribbles_list = reorder_scribbles(scribbles_list)

    print("num of inst captions, masks, boxes and points: ", len(phrases), len(binay_masks), len(locations), len(points))

    # get polygons for each annotation
    polygons_list = []
    segs_list = []
    for idx, mask_fg in enumerate(binay_masks):
        # binary_mask = mask_fg[:,:,0]
        polygons = sample_sparse_points_from_mask(mask_fg, k=256)
        if polygons is None:
            polygons = [0 for _ in range(256*2)]
        polygons = convert_points(polygons, img_info)
        polygons_list.append(polygons)

        segs_list.append(resize(mask_fg.astype(np.float32), (512, 512, 1)))

    segs = np.stack(segs_list).astype(np.float32).squeeze() if len(segs_list) > 0 else segs_list
    polygons = polygons_list
    scribbles = scribbles_list

    meta_list = [ 
        # grounding inputs for generation
        dict(
            ckpt = ckpt,
            prompt = prompt,
            phrases = phrases,
            polygons = polygons,
            scribbles = scribbles,
            segs = segs,
            locations = locations,
            points = points, 
            alpha_type = [args.alpha, 0.0, 1-args.alpha],
            save_folder_name=save_folder_name
        ), 
    ]

    # set seed
    torch.manual_seed(seed)
    starting_noise = torch.randn(args.num_images, 4, 64, 64).to(device)

    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(meta_list[0]["ckpt"], args, device)
    clip_model, clip_processor = create_clip_pretrain_model()

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    for meta in meta_list:
        run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, guidance_scale=args.guidance_scale)