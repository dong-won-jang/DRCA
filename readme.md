# DenseNet with Deep Residual Channel-Attention Blocks for Single Image Super Resolution (DRCA, CVPRw2019)
By Dong-Won Jang and Rae-Hong Park

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN](https://github.com/yulunzhang/RCAN).

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)
6. [Citation](#citation)

## Introduction
### Abstract 

 We have interpreted the role of ResNet (feature value refinement by addition) and DenseNet (feature value memory by concatenation). The contribution of the proposed network is dense connections between residual groups rather than convolution layers. In terms of feature value refinement and memory, the proposed method refines the feature values sufficiently (by residual group) and memorizes the refined feature values intermittently (by dense connections between residual groups).

### Blockdiagram 
![PSNR_SSIM_BI](/Figs/blockdiagram.png)

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or DF2K trainig data [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar). DF2K dataset is merged dataset of DIV2K and Flickr2K.

2. Place your training data as following structure.
traindata/DIV2K or traindata/DF2K.

        .
        DIV2K
        ├── DIV2K_train_HR                   
        │   ├── 0001.png          
        │   ├── 0002.png         
        │   ├── ...
        │   └── 0800.png
        └── DIV2K_train_LR_bicubic               
            ├── X2
            │   ├── 0001x2.png
            │   ├── 0002x2.png
            │   ├── ...
            │   └── 0800x2.png
            ├── X4        
            │   ├── 0001x4.png
            │   ├── 0002x4.png
            │   ├── ...
            │   └── 0800x4.png
            └── ...   

        .
        DF2K
        ├── Train_HR                   
        │   ├── 0001.png          
        │   ├── 0002.png         
        │   ├── ...
        │   └── 3450.png
        └── Train_LR_bicubic               
            ├── X2
            │   ├── 0001x2.png
            │   ├── 0002x2.png
            │   ├── ...
            │   └── 0800x2.png
            ├── X4        
            │   ├── 0001x4.png
            │   ├── 0002x4.png
            │   ├── ...
            │   └── 3450x4.png
            └── ...   
  
Validation images were located at same folder. We use 5 validation images.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

Cd to 'code', run the following scripts to train models.

    **You can use scripts in file 'TrainDRCA_scripts' to train models for our paper.**

    ```bash
    # scale 2, 4 for DIV2K
    # DRCA_BIX2_G5R36P48, input=48x48, output=96x96
    python main.py --model DRCA --save DRCA_BIX2_G5R36P48 --scale 2 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96

    # DRCA_BIX4_G5R36P48, input=48x48, output=192x192
    python main.py --model DRCA --save DRCA_BIX4_G5R36P48 --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/DRCA_BIX2_G5R36P48/model/model_latest.pt

    # scale 4 for DF2K
    # DRCA_BIX4_G5R36P48_DF2K, input=48x48, output=192x192
    python main.py --model DRCA --save DRCA_BIX4_G5R36P48_DF2K --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/DRCA_BIX2_G5R36P48/model/model_latest.pt --data_train DF2K --data_test DF2K --n_train 3450 --offset_val=3450 --test_every 1100
    ```

## Test
### Quick start
1. Place benchmark images as following structure (5 dataset: Set5, Set14, B100, Urban100, Manga109)
Name of HR and LR images should be same except for x2 or x4.

        .
        benchmark
        ├── Set5                   
        │   ├── HR
        │   │   ├── baby.png
        │   │   ├── ...
        │   │   └── woman.png
        │   └── LR_bicubic
        │       ├── X2
        │       │   ├── babyx2.png
        │       │   ├── ...
        │       │   └── womanx2.png
        │       └── X4
        │           ├── babyx4.png
        │           ├── ...
        │           └── womanx4.png
        ├── Set14                   
        │   ├── HR
        │   │   ├── baboon.png
        │   │   ├── ...
        │   │   └── zebra.png
        │   └── LR_bicubic
        │       ├── X2
        │       │   ├── baboonx2.png
        │       │   ├── ...
        │       │   └── zebrax2.png
        │       └── X4
        │           ├── baboonx4.png
        │           ├── ...
        │           └── zebrax4.png
        .
        
2. Download models [GoogleDrive](https://drive.google.com/open?id=15sNruC4Oi6I-trj-HNO5hZOjqPQ_Hhux) for our paper and place them in 'model'.

3. Cd to 'code', run the following scripts to test models.

    **You can use scripts in file 'TestDRCA_scripts' to test models for our paper.**
        
    ```bash
    declare -a dbName=("Set5" "Set14" "B100" "Urban100" "Manga109")
    arrLen=${#dbName[@]}
    scale=4
    trained="DF2K"
    model="DRCA_BIX$scale""_$trained"

    # Large Model
    for ((i=0;i<$arrLen;i++));
    do
        cmd="CUDA_VISIBLE_DEVICES=0 python main.py --self_ensemble --data_test ${dbName[$i]} --scale $scale --model DRCA --n_resgroups 5 --n_resblocks 36 --n_feats 64 --pre_train ../model/$model.pt --test_only --save_results --chop --save 'DRCA_Self_$trained/${dbName[$i]}/X$scale' --testpath ../benchmark"
        eval "$cmd"
    done
    ```

## Results
### Quantitative Results
![PSNR_SSIM_BI](/Figs/result.png)

### Qualitative Results
![Visual_PSNR_SSIM_BI](/Figs/Fig5.png)
![Visual_PSNR_SSIM_BI](/Figs/Fig6.png)
![Visual_PSNR_SSIM_BI](/Figs/Fig7.png)

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN](https://github.com/yulunzhang/RCAN). We thank the authors for sharing their codes.

## Citation

    @InProceedings{Jang_2019_CVPR_Workshops,
    author = {Jang, Dong-Won and Park, Rae-Hong},
    title = {DenseNet With Deep Residual Channel-Attention Blocks for Single Image Super Resolution},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
    }
    

