# Codebase on pre-trained models on Multi-Moments and M4dataset

## A quick command of evaluating the M4dataset pre-trained model

```
python inference.py m4dataset RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 --wd 1e-4 --lr_type cos --epochs 120 --batch-size 36 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb --metadir ./m4dataset/ --loss_type lsep --resume ./checkpoint/TSM_m4dataset_RGB_resnet50_shift8_blockres_avg_segment8_e120_cos_lsep/ckpt.best.pth.tar
```

## Instructions

* Change the dataset root path of Multi-Moments/M4datasets in ```ops/dataset_config.py```.
* Download the pre-trained weights and change the flag ```--resume``` to the weights path.
* First time evaluation will take longer time than usual to generate the video metafile.