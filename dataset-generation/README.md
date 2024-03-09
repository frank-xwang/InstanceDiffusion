
### Pipeline
Dataset preparation consists of three major steps:
1. Image-level label generation
2. Bounding-box and mask generation
3. Instance-level text prompt generation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 2.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
  Note, please check PyTorch version matches that is required by Detectron2.

### Example conda environment setup
Please begin by following the instructions in [INSTALL](https://github.com/frank-xwang/InstanceDiffusion/tree/main?tab=readme-ov-file#installation) to set up the conda environment. After that, proceed with the steps below to install the necessary packages for generating training data.

```bash
# install grounding-sam, ram and grounding-dino
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -U openmim
mim install mmcv
python -m pip install -e GroundingDINO
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
pip install --upgrade diffusers
pip install submitit
# install lavis
pip install salesforce-lavis
pip install webdataset

# download pretrained checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```

### Expected structure for data generation
We should have a 'train_data.json' file with paired image path and image caption as follows. 
Utilizing a subset of the LAION-400M dataset (5~10M images) should enable the reproduction of the results. Additionally, incorporating datasets from GLIGEN could further enhance performance in adhering to bounding box conditions.
```
[
  {
    "image": /PATH/TO/IMAGE1,
    "caption": IMAGE1-CAPTION
  },
  {
    "image": /PATH/TO/IMAGE2,
    "caption": IMAGE2-CAPTION
  }
]
```

### Script for generating training data
We support training data generation with multiple GPUs and nodes. Executing the following commands will produce instance segmentation masks, detection bounding boxes, and instance-level captions for all images listed in `train_data_path`.
```bash
cd dataset-generation/
python run_with_submitit_generate_caption.py \
    --timeout 4000 \
    --partition learn \
    --num_jobs 1 \
    --config ../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --ram_checkpoint ../Grounded-Segment-Anything/ram_swin_large_14m.pth \
    --grounded_checkpoint ../Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint ../Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --box_threshold 0.25 \
    --text_threshold 0.2 \
    --iou_threshold 0.5 \
    --device "cuda" \
    --sam_hq_checkpoint ../Grounded-Segment-Anything/sam_hq_vit_h.pth \
    --use_sam_hq \
    --output_dir "/data/home/xudongw/Grounded-Segment-Anything/sample-data-gen/" \
    --train_data_path train_data.json \
    --output_dir "train-data" \

# For each image, a corresponding JSON file is created in the --output_dir. 
# To compile all file names into a list for model training, use
ls train-data/*.json > train.txt

# Or for handling a large number of files, it is recommended to run
python jsons2txt.py

```
`--num_jobs` specifies how many GPUs are employed for data generation, with jobs automatically distributed across GPUs on multiple machines.
