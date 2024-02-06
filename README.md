# InstanceDiffusion: Instance-level Control for Image Generation

We introduce **InstanceDiffusion** that adds precise instance-level control to text-to-image diffusion models. InstanceDiffusion supports free-form language conditions per instance and allows flexible ways to specify instance locations such as simple **single points**, **scribbles**, **bounding boxes** or intricate **instance segmentation masks**, and combinations thereof. We outperform previous state-of-the-art by 20.4% AP50 for box inputs, and 25.4% IoU for mask inputs.

<p align="center"> <img src='docs/teaser.jpg' align="center" > </p>

> [**InstanceDiffusion: Instance-level Control for Image Generation**](http://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/)            
> [Xudong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Saketh Rambhatla](https://rssaketh.github.io/), 
[Rohit Girdhar](https://rohitgirdhar.github.io/), [Ishan Misra](https://imisra.github.io/)     
> GenAI, Meta; BAIR, UC Berkeley            
> Tech Report            

[[`project page`](http://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/)] [[`arxiv`](https://arxiv.org/abs/2402.03290)] [[`PDF`](https://arxiv.org/pdf/2402.03290.pdf)] [[`bibtex`](#citation)]             


## Disclaimer
This repository represents a re-implementation of InstanceDiffusion conducted by the first author during his time at UC Berkeley. Minor performance discrepancies may exist (differences of ~1% in AP) compared to the results reported in the original paper. The goal of this repository is to replicate the original paper's findings and insights, primarily for academic and research purposes.


## Updates
* 02/05/2024 - Initial commit. Stay tuned


## Installation
### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 2.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
- OpenCV ≥ 4.6 is needed by demo and visualization.

### Conda environment setup
```bash
conda create --name instdiff python=3.8 -y
conda activate instdiff

pip install -r requirements.txt
```


## Training Data Generation
See [Preparing Datasets for InstanceDiffusion](dataset-generation/README.md).


## Method Overview
<p align="center">
  <img src="docs/InstDiff-gif.gif" width=70%>
</p>

InstanceDiffusion enhances text-to-image models by providing additional instance-level control. In additon to a global text prompt, InstanceDiffusion allows for paired instance-level prompts and their locations (e.g. points, boxes, scribbles or instance masks) to be specified when generating images. 
We add our proposed learnable UniFusion blocks to handle the additional per-instance conditioning. UniFusion fuses the instance conditioning with the backbone and modulate its features to enable instance conditioned image generation. Additionally, we propose ScaleU blocks that improve the UNet’s ability to respect instance-conditioning by rescaling the skip-connection and backbone feature maps produced in the UNet. At inference, we propose Multi-instance Sampler which reduces information leakage across multiple instances.

Please check our [paper](https://arxiv.org/abs/xxxx.xxxxx) and [project page](http://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/) for more details.


## InstanceDiffusion Inference Demons
If you want to run InstanceDiffusion demos locally, we provide `inference.py`. Please download the pretrained [InstanceDiffusion](https://drive.google.com/drive/folders/1Jm3bsBmq5sHBnaN5DemRUqNR0d4cVzqG?usp=sharing) and [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt), place them under `pretrained` folder and then run it with:
```
python inference.py \
  --num_images 8 \
  --output OUTPUT/ \
  --input_json demos/demo_cat_dog_robin.json \
  --ckpt pretrained/instancediffusion_sd15.pth \
  --test_config configs/test_box.yaml \
  --guidance_scale 5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.36 \
  --cascade_strength 0.3 \
```
The JSON file `input_json` specifies text prompts and location conditions for generating images, with several demo JSON files available under the `demos` directory. 
The `num_images` parameter indicates how many images to generate. 
The `mis` setting adjusts the proportion of timesteps utilizing multi-instance sampler, recommended to be below 0.4. A higher `mis` value can decrease information leakage between instances and improve image quality, but may also slow the generation process.
The SDXL refiner is activated if the `cascade_strength` is larger than 0. Note: The SDXL-Refiner was not employed for quantitative evaluations in the paper, but we recently found that it can improve the image generation quality.
Adjusting `alpha` modifies the fraction of timesteps using instance-level conditions, where a higher `alpha` ensures better adherence to location conditions at the potential cost of image quality, there is a trade-off.

The bounding box should follow the format [xmin, ymin, width, height]. The mask is expected in RLE (Run-Length Encoding) format. Scribbles should be specified as [[x1, y1],..., [x20, y20]], and a point is denoted by [x, y].


### Image Generation Using Single Points
InstanceDiffusion supports generating images using points (with one point each instance) and corresponding instance captions.
<p align="center">
  <img src="docs/InstDiff-points.png" width=95%>
</p>

```
python inference.py \
  --num_images 8 \
  --output OUTPUT/ \
  --input_json demos/demo_corgi_kitchen.json \
  --ckpt pretrained/instancediffusion_sd15.pth \
  --test_config configs/test_point.yaml \
  --guidance_scale 5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.2 \
  --cascade_strength 0.3 \
```


### Iterative Image Generation
https://github.com/frank-xwang/InstanceDiffusion/assets/58996472/b161455a-6b21-4607-a59d-3a6dd19edab1

InstanceDiffusion can also support iterative image generation, with minimal changes to pre-generated instances and the overall scene. Using the identical initial noise and image caption, InstanceDiffusion can selectively introduce new instances, substitute one instance for another, reposition an instance, or adjust the size of an instance via modifying the bounding boxes. 

```
python inference.py \
  --num_images 8 \
  --output OUTPUT/ \
  --input_json demos/demo_iterative_r1.json \
  --ckpt pretrained/instancediffusion_sd15.pth \
  --test_config configs/test_box.yaml \
  --guidance_scale 5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.2 \
  --cascade_strength 0.3 \
```

`--input_json` can be set to `demo_iterative_r{k+1}.json` for generating images in subsequent rounds.


## Model Evaluation
### Location Conditions (point, scribble, box and instance mask)
coming soon
<p align="center">
  <img src="docs/results.png" width=100%>
</p>

### Attribute Binding
coming soon


## InstanceDiffusion Model Training 
To train InstanceDiffusion with submitit, start by setting up the conda environment according to the instructions in [INSTALL](##Installation). Then, prepare the training data by following the guidelines at [this link](dataset-generation/README.md). Next, download [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt) to the `pretrained` folder. Finally, run the commands below:
```
run_name="instancediffusion"
python run_with_submitit.py \
    --workers 8 \
    --ngpus 8 \
    --nodes 8 \
    --batch_size 8 \
    --base_learning_rate 0.00005 \
    --timeout 20000 \
    --warmup_steps 5000 \
    --partition learn \
    --name=${run_name} \
    --wandb_name ${run_name} \
    --yaml_file="configs/train_sd15.yaml" \
    --official_ckpt_name='pretrained/v1-5-pruned-emaonly.ckpt' \
    --train_file="train.txt" \
    --random_blip 0.5 \
    --count_dup true \
    --use_masked_att true \
    --add_inst_cap_2_global false \
    --enable_ema true \
    --re_init_opt true \
```
For more options, see `python run_with_submitit.py -h`.


## License and Acknowledgment
The majority of InstanceDiffusion is licensed under the [Apache License](LICENSE), however portions of the project are available under separate license terms: CLIP, BLIP, Stable Diffusion and GLIGEN are licensed under their own licenses; If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than Apache, CC-BY-NC, MIT, or CC0.


## Ethical Considerations
InstanceDiffusion's wide range of image generation capabilities may introduce similar challenges to many other text-to-image generation methods. 


## How to get support from us?
If you have any general questions, feel free to email us at [XuDong Wang](mailto:xdwang@eecs.berkeley.edu). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@misc{wang2024instancediffusion,
      title={InstanceDiffusion: Instance-level Control for Image Generation}, 
      author={Xudong Wang and Trevor Darrell and Sai Saketh Rambhatla and Rohit Girdhar and Ishan Misra},
      year={2024},
      eprint={2402.03290},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
