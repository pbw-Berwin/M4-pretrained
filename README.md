# Codebase on pre-trained models on Multi-moments and M4 dataset

## Overview
This is the repo for the TSM ResNet50 model trained on the M4 dataset (Multi-Moments minus Memento) first introduced in the BOLD Moments Dataset (BMD) manuscript.
Model weights and extracted features for BMD's 1,102 video stimulus set are available for download in BMD's OpenNeuro repository [https://openneuro.org/datasets/ds005165] in the filepath: ./derivatives/Temporal_Shift_Module_DNN/

Here is a relevant exercpt from BOLD Moments manuscript describing the model training:
>The model adopts the architecture of a Temporal Shift Module (TSM) (Lin et al., 2019), with ResNet50 as the backbone network. We trained our model on the M4 (Multi-Moments minus Memento) training dataset for 120 epochs by using LSEP (log-sum-exp pairwise) loss (Monfort et al., 2022). LSEP loss was first proposed in (Li et al., 2017) and modified in (Monfort et al., 2022) as an appropriate loss function to train on multi-label and class imbalanced datasets, such as actions. The M4 training dataset consists of 1,012,169 videos which are in the Multi-Moments in Time dataset but not in the Memento dataset to ensure no overlap with the 1,102 BMD stimuli. Our model was initialized with the weights of the ResNet50 trained on ImageNet-1k dataset. We chose the model hyperparameters to closely follow those used in Lin and colleagues (2019). Specifically, during the training phase, our model split the input video into 8 segments and sampled 1 frame from each segment. We used SGD optimizer to optimize our model. The learning rate followed the cosine learning rate schedule and was initialized as 0.02. The weight decay was set to be 0.0001 and the batch size 128. The model achieved a precision-at-one score 0.593, a precision-at-five score of 0.829, and a mAP score of 0.636 (loss of 2.75054). Model training took 3 months on 16 V100 GPUs.

## A quick command of evaluating the M4-dataset pre-trained model

```
python inference.py m4dataset RGB --arch resnet50 --num_segments 8 --batch-size 36 -j 32 --consensus_type=avg --shift --shift_div=8 --shift_place=blockres --npb --metadir ./m4dataset/ --loss_type lsep --resume /Path/To/The/Pretrained/Weights
```

Example: ```python inference.py m4dataset RGB --arch resnet50 --num_segments 8 --batch-size 36 -j 32 --consensus_type=avg --shift --shift_div=8 --shift_place=blockres --npb --metadir ./m4dataset/ --loss_type lsep --resume ./checkpoint/TSM_m4dataset_RGB_resnet50_shift8_blockres_avg_segment8_e120_cos_lsep/ckpt.pth.tar```

## Instructions

* Change the dataset root path of Multi-Moments/M4datasets in ```ops/dataset_config.py```.
* Download the pre-trained weights and change the flag ```--resume``` to the weights path.
* First time evaluation will take longer time than usual to generate the video metafile.

## Download pretrained weights and extracted BMD features
```
LOCAL_DIR="/your/path/to/BOLDMomentsDataset"
mkdir -p "${LOCAL_DIR}/derivatives/Temporal_Shift_Module_DNN"

aws s3 cp --no-sign-request --recursive \
"s3://openneuro.org/ds005165/derivatives/Temporal_Shift_Module_DNN/" \
"${LOCAL_DIR}/derivatives/Temporal_Shift_Module_DNN/"
```

## Citation
If you use this model, please consider citing the BOLD Moments paper:

CITATION TODO
