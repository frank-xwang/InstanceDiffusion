def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False

def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)

def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def get_trainable_params(model, original_params_names):
    params = []
    trainable_names = []
    all_params_name = []
    for name, p in model.named_parameters():
        if ("transformer_blocks" in name) and ("fuser" in name):
            # New added Attention layers 
            params.append(p) 
            trainable_names.append(name)
        elif  "position_net" in name:
            # Grounding token processing network 
            params.append(p) 
            trainable_names.append(name)
        elif  "downsample_net" in name:
            # Grounding downsample network (used in input) 
            params.append(p) 
            trainable_names.append(name)
        elif 'scaleu' in name:
            params.append(p) 
            trainable_names.append(name)
        else:
            # Following make sure we do not miss any new params
            # all new added trainable params have to be haddled above
            # otherwise it will trigger the following error  
            assert name in original_params_names, name 
        all_params_name.append(name) 
    print("Trainable params: ", trainable_names)
    return params