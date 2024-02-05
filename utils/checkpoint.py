import io
import os
import torch
import torchvision
from .dist import get_rank
from omegaconf import OmegaConf
from torch import distributed as dist
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.jsondataset import sub_batch, batch_to_device

def read_official_ckpt(ckpt_path):
    "Read offical pretrained SD ckpt and convert into our style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint

class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, masked_real, captions, seen, batch=None):
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 


def load_autoresume_ckpt(checkpoint, config, model, ema, opt, scheduler):
    starting_iter = 0  
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        # assert unexpected_keys == [] and missing_keys == [], "missing keys in pretrained model: {}, unexpected_keys keys in pretrained model: {}".format(missing_keys, unexpected_keys)
        print("missing keys in pretrained model: {}".format(missing_keys))
        print("unexpected keys in pretrained model: {}".format(unexpected_keys))

        if config.enable_ema:
            ema.load_state_dict(checkpoint["ema"], strict=False)
        if not config.re_init_opt:
            opt.load_state_dict(checkpoint["opt"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        starting_iter = checkpoint["iters"]
        if starting_iter >= config.total_iters:
            synchronize()
            print("Training finished. Start exiting")
            exit()

    return starting_iter


@torch.no_grad()
def save_ckpt(config, model, text_encoder, autoencoder, opt, scheduler, config_dict, diffusion, ema, iter_idx, name):
    if get_rank() == 0:
        model_wo_wrapper = model.module if config.distributed else model
        ckpt = dict(model = model_wo_wrapper.state_dict(),
            text_encoder = text_encoder.state_dict(),
            autoencoder = autoencoder.state_dict(),
            opt = opt.state_dict(),
            scheduler = scheduler.state_dict(),
            iters = iter_idx+1,
            config_dict = config_dict,
        )
        ckpt['diffusion'] = diffusion.state_dict()
        if config.enable_ema:
            ckpt["ema"] = ema.state_dict()
        torch.save( ckpt, os.path.join(name, "checkpoint_latest.pth") )


@torch.no_grad()
def save_ckpt_and_result(config, model, text_encoder, autoencoder, opt, scheduler, config_dict, diffusion, ema, iter_idx, loader_train, dataset_train, grounding_tokenizer_input, image_caption_saver, name, device):
    if get_rank() == 0:
        model_wo_wrapper = model.module if config.distributed else model
        iter_name = iter_idx + 1 # we use iter_idx + 1 as the checkpoint name

        if not config.disable_inference_in_training:
            # Do an inference on one training batch
            batch_here = config.batch_size
            batch_num = 0
            # we save result for 10 iters for visualization and debugging purpose
            for idx, batch in enumerate(loader_train):
                if batch_num >= 10:
                    break
                iter_name = iter_idx + 1 + idx
                batch = sub_batch(batch, batch_here)
                batch_to_device(batch, device)

                if "boxes" in batch:
                    real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                    for i in range(batch_here):
                        temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                        im = dataset_train.decode_func.vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                        real_images_with_box_drawing.append(im)
                    real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
                else:
                    # keypoint case 
                    real_images_with_box_drawing = batch["image"]*0.5 + 0.5 

                uc = text_encoder.encode( batch_here*[""] )
                context = text_encoder.encode(  batch["caption"]  )

                # check if self.diffusion has config attribute and prediction_type is v_prediction
                if hasattr(diffusion, 'config') and diffusion.config.prediction_type == "v_prediction":
                    plms_sampler = diffusion
                else:
                    plms_sampler = PLMSSampler(diffusion, model_wo_wrapper)

                shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)

                grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=config.use_masked_att)
                input = dict( x=None, 
                            timesteps=None, 
                            context=context, 
                            grounding_input=grounding_input )
                samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)

                autoencoder_wo_wrapper = autoencoder # Note itself is without wrapper since we do not train that. 
                samples = autoencoder_wo_wrapper.decode(samples).cpu()
                samples = torch.clamp(samples, min=-1, max=1)

                masked_real_image =  None
                image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, batch["caption"], iter_name, batch)
                batch_num += 1

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = text_encoder.state_dict(),
                    autoencoder = autoencoder.state_dict(),
                    opt = opt.state_dict(),
                    scheduler = scheduler.state_dict(),
                    iters = iter_idx+1,
                    config_dict = config_dict,
        )
        ckpt['diffusion'] = diffusion.state_dict()
        if config.enable_ema:
            ckpt["ema"] = ema.state_dict()
        torch.save( ckpt, os.path.join(name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(name, "checkpoint_latest.pth") )

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def load_model_ckpt(ckpt_path, args, device):
    saved_ckpt = torch.load(ckpt_path)
    if hasattr(args, 'test_config') and args.test_config != "":
        config = OmegaConf.load(args.test_config) 
        config = vars(config)["_content"]
        print("config for evaluation: ", config)
    else:
        config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    try:
        # load ema model if exists
        print("Loading ema")
        model.load_state_dict( saved_ckpt['ema'] )
    except:
        print("Loading non-ema model")
        model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"], strict=False )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config