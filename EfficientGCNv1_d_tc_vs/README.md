# Efficient Graph Convolutional Network (EfficientGCN) (ResGCNv2.0)

## 1 Paper Details

Yi-Fan Song, Zhang Zhang, Caifeng Shan and Liang Wang. **EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition**. Submitted to IEEE T-PAMI, 2021. [[Arxiv Preprint]](https://arxiv.org/pdf/2106.15125.pdf)

Previous version (ResGCN v1.0): [[Github]](https://github.com/yfsong0709/ResGCNv1) [[ACMMM 2020]](https://dl.acm.org/doi/abs/10.1145/3394171.3413802) [[Arxiv Preprint]](https://arxiv.org/pdf/2010.09978.pdf)

The following pictures are the pipeline of EfficientGCN and the illustration of ST-JointAtt layer, respectively.
<div align="center">
    <img src="resources/pipeline.jpg">
</div>

<div align="center">
    <img src="resources/st_joint_att.jpg">
</div>
			neighbor_link = [(1, 5), 
			(12, 5), (2, 12), (3, 2), 
							 (11, 5), (10, 11), (6, 10),
                             (8, 1), (13, 1), 
							 (9, 5), (7, 9),(4,9)]
            connect_joint = np.array([5, 12, 2, 9, 5,
                                      10, 9, 1,
                                      5, 11, 5, 5,1])  # 12
            parts = [
                np.array([11, 10, 6]),  # left_arm
                np.array([12, 2, 3]),  # right_arm
                np.array([1, 5])  # torso
            ]

## 2 Prerequisites

### 2.1 Libraries

This code is based on [Python3](https://www.anaconda.com/) (anaconda, >=3.5) and [PyTorch](http://pytorch.org/) (>=1.6.0).

Other Python libraries are presented in the **'scripts/requirements.txt'**, which can be installed by 
```
pip install -r scripts/requirements.txt
```

### 2.2 Experimental Dataset

Our models are experimented on the **NTU RGB+D 60 & 120** datasets, which can be downloaded from 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

There are 302 samples of **NTU RGB+D 60** and 532 samples of **NTU RGB+D 120** need to be ignored, which are shown in the **'src/preprocess/ignore.txt'**.

### 2.3 Pretrained Models

Several pretrained models are provided, which include **EfficientGCN-B0**, **EfficientGCN-B2**, and **EfficientGCN-B4** for the **cross-subject (X-sub)** and **cross-view (X-view)** benchmarks of the **NTU RGB+D 60** dataset and the **cross-subject (X-sub120)** and **cross-setup (X-set120)** benchmarks of the **NTU RGB+D 120** dataset.

These models can be downloaded from [BaiduYun]() (Extraction code: **1c5x**) or [GoogleDrive](https://drive.google.com/drive/folders/1HpvkKyfmmOCzuJXemtDxQCgGGQmWMvj4?usp=sharing).


## 3 Parameters

Before training and evaluating, there are some parameters should be noticed.

* (1) **'--config'** or **'-c'**: The config of EfficientGCN. You must use this parameter in the command line or the program will output an error. There are 12 configs given in the **configs** folder, which can be illustrated in the following tabel.

| config    | 2001   | 2002   | 2003     | 2004     |
| :-------: | :----: | :----: | :------: | :------: |
| model     | B0     | B0     | B0       | B0       |
| benchmark | X-sub  | X-view | X-sub120 | X-set120 |

| config    | 2005   | 2006   | 2007     | 2008     |
| :-------: | :----: | :----: | :------: | :------: |
| model     | B2     | B2     | B2       | B2       |
| benchmark | X-sub  | X-view | X-sub120 | X-set120 |

| config    | 2009   | 2010   | 2011     | 2012     |
| :-------: | :----: | :----: | :------: | :------: |
| model     | B4     | B4     | B4       | B4       |
| benchmark | X-sub  | X-view | X-sub120 | X-set120 |

* (2) **'--work_dir'** or **'-w'**: The path to workdir, for saving checkpoints and other running files. Default is **'./workdir'**.

* (3) **'--pretrained_path'** or **'-pp'**: The path to the pretrained models. **pretrained_path = None** means using randomly initial model. Default is **None**.

* (4) **'--resume'** or **'-r'**: Resume from the recent checkpoint (**'<--work_dir>/checkpoint.pth.tar'**).

* (5) **'--evaluate'** or **'-e'**: Only evaluate models. You can choose the evaluating model according to the instructions.

* (6) **'--extract'** or **'-ex'**: Extract features from a trained model for visualization. Using this parameter will make a data file named **'extraction_<--config>.npz'** at the **'./visualization'** folder.

* (7) **'--visualization'** or **'-v'**: Show the information and details of a trained model. You should extract features by using **<--extract>** parameter before visualizing.

* (8) **'--dataset'** or **'-d'**: Choose the dataset. (Choice: **[ntu-xsub, ntu-xview, ntu-xsub120, ntu-xset120]**)

* (9) **'--model_type'** or **'-mt'**: Choose the model. (Format: **EfficientGCN-B{coefficient}**, e.g., EfficientGCN-B0, EfficientGCN-B2, EfficientGCN-B4)

Other parameters can be updated by modifying the corresponding config file in the **'configs'** folder or using command line to send parameters to the model, and the parameter priority is **command line > yaml config > default value**.


## 4 Running

### 4.1 Modify Configs

Firstly, you should modify the **'path'** parameters in all config files of the **'configs'** folder.

A python file **'scripts/modify_configs.py'** will help you to do this. You need only to change three parameters in this file to your path to NTU datasets.
```
python scripts/modify_configs.py --path <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>
```
All the commands above are optional.

### 4.2 Generate Datasets

After modifing the path to datasets, please generate numpy datasets by using **'scripts/auto_gen_data.sh'**.
```
bash scripts/auto_gen_data.sh
```
It may takes you about 2.5 hours, due to the preprocessing module for X-view benchmark (same as [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)).

To save time, you can only generate numpy data for one benchmark by the following command (only the first time to use this benchmark).
```
python main.py -c <config> -gd
```
where `<config>` is the config file name in the **'configs'** folder, e.g., 2001.

**Note:** only training the X-view benchmark requires preprocessing module.

### 4.3 Train

You can simply train the model by 
```
python main.py -c <config>
```
If you want to restart training from the saved checkpoint last time, you can run
```
python main.py -c <config> -r
```

### 4.4 Evaluate

Before evaluating, you should ensure that the trained model corresponding the config is already existed in the **<--pretrained_path>** or **'<--work_dir>'** folder. Then run
```
python main.py -c <config> -e
```

### 4.5 Visualization

To visualize the details of the trained model, you can run
```
python main.py -c <config> -ex -v
```
where **'-ex'** can be removed if the data file **'extraction_`<config>`.npz'** already exists in the **'./visualization'** folder.


## 5 Results

Top-1 Accuracy for the provided models on **NTU RGB+D 60 & 120** datasets.

| models          | FLOPs  | parameters | NTU X-sub  | NTU X-view | NTU X-sub120 | NTU X-set120 |
| :-------------: | :----: | :--------: | :--------: | :--------: | :----------: | :----------: |
| EfficientGCN-B0 | 3.08G  | 0.32M      | 89.9%      | 94.7%      | 85.9%        | 84.3%        |
| EfficientGCN-B2 | 6.12G  | 0.79M      | 90.9%      | 95.5%      | 87.9%        | 88.0%        |
| EfficientGCN-B4 | 15.24G | 2.03M      | **91.7%**  | **95.7%**  | **88.3%**    | **89.1%**    |

## 6 Citation and Contact

If you have any question, please send e-mail to `yifan.song@cripac.ia.ac.cn`.

Please cite our paper when you use this code in your research.
```
@article{song2021efficientgcn,
  author    = {Song, Yi-Fan and Zhang, Zhang and Shan, Caifeng and Wang, Liang},
  title     = {EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition},
  journal   = {arXiv:2106.15125},
  year      = {2021},
}

@inproceedings{song2020stronger,
  author    = {Song, Yi-Fan and Zhang, Zhang and Shan, Caifeng and Wang, Liang},
  title     = {Stronger, Faster and More Explainable: A Graph Convolutional Baseline for Skeleton-Based Action Recognition},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (ACMMM)},
  pages     = {1625--1633},
  year      = {2020},
  isbn      = {9781450379885},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3394171.3413802},
  doi       = {10.1145/3394171.3413802},
}
```
