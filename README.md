# Codebase on pre-trained models on Multi-moments and M4 dataset

## A quick command of evaluating the M4-dataset pre-trained model

```
python inference.py m4dataset RGB --arch resnet50 --num_segments 8 --batch-size 36 -j 32 --consensus_type=avg --shift --shift_div=8 --shift_place=blockres --npb --metadir ./m4dataset/ --loss_type lsep --resume /Path/To/The/Pretrained/Weights
```

Example: ```python inference.py m4dataset RGB --arch resnet50 --num_segments 8 --batch-size 36 -j 32 --consensus_type=avg --shift --shift_div=8 --shift_place=blockres --npb --metadir ./m4dataset/ --loss_type lsep --resume ./checkpoint/TSM_m4dataset_RGB_resnet50_shift8_blockres_avg_segment8_e120_cos_lsep/ckpt.pth.tar```

## Instructions

* Change the dataset root path of Multi-Moments/M4datasets in ```ops/dataset_config.py```.
* Download the pre-trained weights and change the flag ```--resume``` to the weights path.
* First time evaluation will take longer time than usual to generate the video metafile.