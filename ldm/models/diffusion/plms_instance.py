import torch
import numpy as np
from copy import deepcopy
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps


class PLMSSamplerInst(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None, mis=0.0):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale
        self.mis = mis

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None):
        self.make_schedule(ddim_num_steps=S)
        return self.plms_sampling(shape, input, uc, guidance_scale, mask=mask, x0=x0)


    @torch.no_grad()
    def plms_sampling(self, shape, input_all, uc=None, guidance_scale=1, mask=None, x0=None):

        b = shape[0]
        latent_size = shape[2]

        img = input_all[0]["x"]
        if img == None:
            img = torch.randn(shape, device=self.device)
            for input_inst in input_all:
                input_inst["x"] = img

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(time_range))

        mis_step = int(total_steps * self.mis)
        # print("stop mergeing at: {}/{} - {}".format(mis_step, len(time_range), time_range))

        old_eps_dic = {}
        for input_idx, input in enumerate(input_all):
            old_eps_dic[str(input_idx)] = []
            for i, step in enumerate(time_range[:mis_step]):
                # set alpha and restore first conv layer 
                if self.alpha_generator_func != None:
                    self.set_alpha_scale(self.model, alphas[i])
                    if alphas[i] == 0:
                        self.model.restore_first_conv_from_SD()

                # run 
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=self.device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)

                img, _, e_t = self.p_sample_plms(input, ts, index=index, uc=uc, guidance_scale=guidance_scale, old_eps=old_eps_dic[str(input_idx)], t_next=ts_next)
                input["x"] = img
                old_eps_dic[str(input_idx)].append(e_t)
                if len(old_eps_dic[str(input_idx)]) >= 4:
                    old_eps_dic[str(input_idx)].pop(0)

        # print("mergeing instances")

        input = input_all[0] # use full-conditioning latent as base
        old_eps = old_eps_dic[str(0)]

        # crop tensor using a bounding box
        def crop_tensor(tensor, bbox):
            return tensor[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]

        # get bounding box of the input
        def get_bbox(input):
            bbox = input["grounding_input"]["boxes"].cpu().numpy()[0][0]
            bbox = [int(x*latent_size) for x in bbox]
            return bbox

        # replace partial tensor with a cropped one
        def crop_and_paste_tensor(target_tensor, source_input):
            bbox = get_bbox(source_input)
            source_tensor = crop_tensor(source_input["x"], bbox)
            target_tensor[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0.0*target_tensor[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] + 1.0*source_tensor
            return target_tensor

        crop_and_paste_latents = False
        if crop_and_paste_latents:
            # crop and paste tensors
            for source_input in input_all[1:]:
                input["x"] = crop_and_paste_tensor(input_all[0]["x"], source_input)
        else:
            # or simply average (default choice)
            input["x"] = torch.mean(torch.stack([input["x"] for input in input_all[:]]), dim=0)

        # remaining timesteps are unchanged
        for i, step in enumerate(time_range):
            if i < mis_step:
                continue
            # set alpha and restore first conv layer
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[i])
                if  alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            # run 
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)

            img, pred_x0, e_t = self.p_sample_plms(input, ts, index=index, uc=uc, guidance_scale=guidance_scale, old_eps=old_eps, t_next=ts_next)
            input["x"] = img
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

        return img


    @torch.no_grad()
    def p_sample_plms(self, input, t, index, guidance_scale=1., uc=None, old_eps=None, t_next=None):
        x = deepcopy(input["x"]) 
        b = x.shape[0]

        def get_model_output(input):
            e_t = self.model(input) 
            if uc is not None and guidance_scale != 1:
                unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc)
                e_t_uncond = self.model( unconditional_input ) 
                e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            return e_t


        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        input["timesteps"] = t 
        e_t = get_model_output(input)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            input["x"] = x_prev
            input["timesteps"] = t_next
            e_t_next = get_model_output(input)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
