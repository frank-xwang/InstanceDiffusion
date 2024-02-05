import time 
import torch
from dataset.jsondataset import batch_to_device
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist import get_rank
from utils.scheduler import create_scheduler
from utils.dataloader import create_dataloader
from utils.misc import AverageMeter, ProgressMeter, sec_2_hms, save_config
from utils.optimizer import get_trainable_params, count_params, update_ema
from utils.model import create_model, load_ckpt, create_ema, create_grounding_tokenizer
from utils.checkpoint import create_expt_folder_with_auto_resuming, ImageCaptionSaver, load_autoresume_ckpt, save_ckpt, save_ckpt_and_result

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")

        # create output folder
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        self.config_dict = save_config(config, self.name) if get_rank() == 0 else None

        # create model and diffusion
        self.model, self.autoencoder, self.text_encoder, self.diffusion, self.text_model, self.text_processor = create_model(config, self.device)

        # load pretrained model
        original_params_names = load_ckpt(config, self.model, self.autoencoder, self.text_encoder, self.diffusion)

        # create optimizer
        params = get_trainable_params(self.model, original_params_names)
        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)

        # create EMA model
        self.ema, self.ema_params, self.master_params = create_ema(self.model, config.enable_ema)

        # create scheduler
        self.scheduler = create_scheduler(config, self.opt)

        # create dataloader
        self.dataset_train, self.loader_train = create_dataloader(config)

        # load from autoresuming ckpt
        self.starting_iter = load_autoresume_ckpt(checkpoint, config, self.model, self.ema, self.opt, self.scheduler)

        # create grounding inputs tokenizer
        self.grounding_tokenizer_input = create_grounding_tokenizer(config, self.model)

        self.image_caption_saver = ImageCaptionSaver(self.name) if get_rank() == 0 else None

        if config.distributed:
            # http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )


    def train_one_epoch(self, epoch, total_epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(self.loader_train),
            [batch_time, data_time, losses], 
            prefix="Epoch: [{}|{}]".format(epoch+1, total_epoch))

        end = time.time()

        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

        self.model.train()
        for iter_idx, batch in enumerate(self.loader_train):

            # measure data loading time
            data_time.update(time.time() - end)

            self.opt.zero_grad(set_to_none=True)
            batch_to_device(batch, self.device)

            # forward
            loss = self.run_one_step(batch)
            if not torch.isnan(loss):
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(self.opt)

                # Updates the scale for next iteration.
                scaler.update()

                # update scheduler
                self.scheduler.step()

                # update ema model
                if self.config.enable_ema:
                    update_ema(self.ema_params, self.master_params, self.config.ema_rate)

                # record loss
                losses.update(loss.item())
                if (get_rank() == 0):
                    if (iter_idx % 10 == 0):
                        self.log_loss()
            else:
                print("nan loss encountered, skipping this batch")

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress
            print_freq = 10
            if iter_idx % print_freq == 0:
                secs = batch_time.avg * (self.config.total_iters - self.iter_idx)
                progress.display(iter_idx, lr=self.opt.param_groups[0]['lr'], remaining_time=sec_2_hms(int(secs)))

            self.iter_idx += 1

            # save ckpt as checkpoint_latest.pth every 2000 iters
            if self.iter_idx % 2000 == 0:
                save_ckpt(self.config, self.model, self.text_encoder, self.autoencoder, self.opt, self.scheduler, self.config_dict, self.diffusion, self.ema, self.iter_idx, self.name)
            # save ckpt and results every save_every_iters iters
            if self.iter_idx % self.config.save_every_iters == 0:
                save_ckpt_and_result(self.config, self.model, self.text_encoder, self.autoencoder, self.opt, self.scheduler, self.config_dict, self.diffusion, self.ema, self.iter_idx, self.loader_train, self.dataset_train, self.grounding_tokenizer_input, self.image_caption_saver, self.name, self.device)


    def start_training(self):
        self.config.total_iters = self.config.total_epochs * len(self.loader_train)
        self.iter_idx = self.starting_iter
        start_epoch = self.starting_iter // len(self.loader_train)
        # training loop
        for epoch in range(start_epoch, self.config.total_epochs):
            if self.config.distributed:
                self.loader_train.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, self.config.total_epochs)

        # save the final ckpt and result
        if get_rank() == 0:
            save_ckpt_and_result(self.config, self.model, self.text_encoder, self.autoencoder, self.opt, self.scheduler, self.config_dict, self.diffusion, self.ema, self.iter_idx, self.loader_train, self.dataset_train, self.grounding_tokenizer_input, self.image_caption_saver, self.name, self.device)
        print("Model training is completed!!!")


    @torch.no_grad()
    def get_input(self, batch):
        z = self.autoencoder.encode( batch["image"] )
        noise = torch.randn_like(z)

        context = self.text_encoder.encode( batch["caption"] )
        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000 ).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        return z, noise, t, context


    def run_one_step(self, batch):
        x_start, noise, t, context = self.get_input(batch)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        
        grounding_input = self.grounding_tokenizer_input.prepare(batch, return_att_masks=self.config.use_masked_att)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    grounding_input=grounding_input)

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if self.config.fp32:
            model_output = self.model(input) # model output: epsilon_t
            target = noise
            loss = torch.nn.functional.mse_loss(model_output, target)
        else:
            with torch.cuda.amp.autocast():
                model_output = self.model(input)
                target = noise
                loss = torch.nn.functional.mse_loss(model_output, target)
        self.loss_dict = {"loss": loss.item()}
        return loss


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )
