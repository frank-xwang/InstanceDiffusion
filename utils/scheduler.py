from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

def create_scheduler(config, opt):
    if config.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
    elif config.scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=config.warmup_steps)
    else:
        assert False 
    return scheduler