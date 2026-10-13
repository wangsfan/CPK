The implementation of the paper *Weakly-supervised incremental learning for semantic segmentation via class proximal knowledge transfer. Displays, 95, 2026, 103627:1-14. Authors: Ning Zhu, Huilin Shi, Jingyi Wang, Shaofan Wang, Yanfeng Sun*. Cite this paper if you use the code
```
@article{ZHU2026weakly,
title = {Weakly-supervised incremental learning for semantic segmentation via class proximal knowledge transfer},
journal = {Displays},
volume = {95},
pages = {103627:1-14},
year = {2026},
author = {Ning Zhu and Huilin Shi and Jingyi Wang and Shaofan Wang and Yanfeng Sun},}
```
## How to run
### Requirements
We have simple requirements:
The main requirements are:
```
python > 3.1
pytorch > 1.6
```
If you want to install a custom environment for this code, you can run the following using [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html):
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tensorboard
conda install jupyter
conda install matplotlib
conda install tqdm
conda install imageio

pip install inplace-abn # this should be done using CUDA compiler (same version as pytorch)
pip install wandb # to use the WandB logger
```

### Datasets 
In the benchmark there are two datasets: Pascal-VOC 2012 and COCO (object only).
For the COCO dataset, we followed the COCO-stuff splits and annotations, that you can see [here](https://github.com/nightrome/cocostuff/).

To download dataset, follow the scripts: `data/download_voc.sh`, `data/download_coco.sh` 

To use the annotations of COCO-Stuff in our setting, you should preprocess it by running the provided script. \
Please, remember to change the path in the script before launching it!
`python data/coco/make_annotation.py`


If your datasets are in a different folder, make a soft-link from the target dataset to the data folder.
We expect the following tree:
```
data/voc/
    SegmentationClassAug/
        <Image-ID>.png
    JPEGImages/
        <Image-ID>.png
    split/
    ... other files 
    
data/coco/
    annotations/
        train2017/
            <Image-ID>.png
        val2017/
            <Image-ID>.png
    images/
        train2017/
            <Image-ID>.png
        val2017/
            <Image-ID>.png
    ... other files 
```
:warning: Bee sure not to override the current `voc` directory of the repository. 
We suggest to link the folders inside the voc directory.

Finally, to prepare the COCO-to-VOC setting we need to map the VOC labels into COCO. Do that by running
`python data/make_cocovoc.py`


### ImageNet Pretrained Models
After setting the dataset, you download the models pretrained on ImageNet using [InPlaceABN](https://github.com/mapillary/inplace_abn).
[Download](https://drive.google.com/file/d/1rQd-NoZuCsGZ7_l_X9GO1GGiXeXHE8CT/view) the ResNet-101 model (we only need it but you can also [download other networks](https://github.com/mapillary/inplace_abn) if you want to change it).
Then, put the pretrained model in the `pretrained` folder.

### Run!
We provide different an example script to run the experiments (see `run.sh`, `coco.sh`).
In the following, we describe the basic parameter to run an experiment.
First, we assume that we have a command 
```
exp='python -m torch.distributed.launch --nproc_per_node=<num GPUs> --master_port <PORT> run.py --num_workers <N_Workers>'`
```
that allow us to setup the distributed data parallel script.

The first to replicate us, is to obtain the model on the step 0 (base step, fully supervised). You can run:
```
exp --name Base --step 0 --lr 0.01 --bce --dataset <dataset> --task <task> --batch_size 24 --epochs 30 --val_interval 2 [--overlap]
```
where we use `--bce` to train the classifier with the binary cross-entropy. `dataset` can be `voc` or `coco-voc`. The task 
are, 
```
voc: (you can set overlap here)
    15-5, 10-10
coco: (overlap is not used)
    voc 
```

After this, you can run the incremental steps using only image level labels (set the `weakly` parameter).
```
exp --name ours --step 1 --weakly --lr 0.001 --alpha 0.5 --step_ckpt <pretr> --loss_de 1 --lr_policy warmup --affinity \ 
    --dataset <dataset> --task <task> --batch_size 24 --epochs 40 [--overlap]
```
where `pretr` should be the path to the pretrained model (usually `checkpoints/step/<dataset>-<task>/<name>.pth`). 
Please, set `--alpha 0.9` on the COCO dataset.
