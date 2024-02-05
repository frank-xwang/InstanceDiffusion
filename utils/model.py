import os
import torch
import numpy as np 
from copy import deepcopy
from PIL import Image, ImageDraw
from .checkpoint import read_official_ckpt
from .optimizer import disable_grads
from ldm.util import instantiate_from_config
from transformers import CLIPProcessor, CLIPModel
from ldm.modules.attention import GatedSelfAttentionDense

def create_clip_pretrain_model():
    # get text model and processer for text
    text_version = "openai/clip-vit-large-patch14"
    text_model = CLIPModel.from_pretrained(text_version).cuda().eval()
    disable_grads(text_model)
    text_processor = CLIPProcessor.from_pretrained(text_version)
    return text_model, text_processor

def create_model(config, device):
    model = instantiate_from_config(config.model).to(device)
    autoencoder = instantiate_from_config(config.autoencoder).to(device)
    text_encoder = instantiate_from_config(config.text_encoder).to(device)
    ## use DDIM noise scheduler
    diffusion = instantiate_from_config(config.diffusion).to(device)

    # get text model and processer
    text_model, text_processor = create_clip_pretrain_model()

    return model, autoencoder, text_encoder, diffusion, text_model, text_processor

def load_ckpt(config, model, autoencoder, text_encoder, diffusion):
    # load pretrained model
    print("ckpt loaded from: ", os.path.join(config.DATA_ROOT, config.official_ckpt_name))        
    state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )

    missing_keys, unexpected_keys = model.load_state_dict( state_dict["model"], strict=False  )
    assert unexpected_keys == []
    print("missing_keys: ", missing_keys)
    original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 

    autoencoder.load_state_dict( state_dict["autoencoder"]  )
    text_encoder.load_state_dict( state_dict["text_encoder"], strict=False )
    diffusion.load_state_dict( state_dict["diffusion"]  )

    autoencoder.eval()
    text_encoder.eval()
    disable_grads(autoencoder)
    disable_grads(text_encoder)

    if config.ckpt is not None:
        first_stage_ckpt = torch.load(config.ckpt, map_location="cpu")
        model.load_state_dict(first_stage_ckpt["model"])

    return original_params_names

def create_ema(model, enable_ema=False):
    if enable_ema:
        master_params = list(model.parameters()) 
        ema = deepcopy(model)
        ema_params = list(ema.parameters())
        ema.eval()
    else:
        ema = None
        ema_params = None
        master_params = None
    return ema, ema_params, master_params

def create_grounding_tokenizer(config, model):
    # func return input for grounding tokenizer 
    if isinstance(config, dict):
        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    else:
        grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
    model.grounding_tokenizer_input = grounding_tokenizer_input
    return grounding_tokenizer_input

def set_alpha_scale(model, alpha_scale):
    for module in model.modules():
        if type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale

def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda() # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
        feature = ( feature / feature.norm() )  * 28.7 
        feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        feature = outputs.text_model_output.pooler_output
    return feature
