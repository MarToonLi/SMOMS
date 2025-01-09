这一版同样起个代号：麻雀，意为麻雀小而全。
2023年4月16日：
- 删除了drive&Act以及DAD数据集的配置文件。因此该项目EfficientGCNv1_d_tc仅面向3MDAD和EBDD两个数据集。
- 配置文件中参数的说明：一般改动不是通过修改yaml(除了与数据集相关的root_folder)，而是通过修改sh文件中的python命令行参数。
  - 命令行参数(优先级最大)、yaml文件参数值（优先级次之）、argparse中的默认参数值。
  - 命令行参数，一般通过sh文件启动，其中scripts中的lingshen.sh和yishen.sh。
  - 一审意见修改期间，该项目已经无法复现到RGB1准确率90%，RGB2准确率86.25%和EBDD准确率96.97%的水平。现已经变为（86.88，85.62，93.94） 
  - 而实验无法复现的理由似乎是由于以下原因：[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
    但是现在无法修复该问题。
  - 实验的硬件配置是：Tesla T4，软件环境是：Pytorch 1.7.1、cuda 10.1、cudnn7.6.5.32、Anaconda2021.05.
    其中python虚拟环境：python3.7.11、安装库如下所示，其中安装⭐标记的库，即可将其他的库连带安装。
    ----------------- --------------------
    - certifi           2022.12.7
    - cycler            0.11.0
    - fonttools         4.38.0
    - kiwisolver        1.4.4
    - matplotlib        3.5.3  (⭐)
    - numpy             1.21.6
    - packaging         23.1
    - pandas            1.3.5
    - Pillow            9.5.0
    - pip               22.3.1
    - pynvml            11.5.0  (⭐)
    - pyparsing         3.0.9
    - python-dateutil   2.8.2
    - pytz              2023.3
    - PyYAML            6.0
    - seaborn           0.12.2  (⭐)
    - setuptools        65.6.3
    - six               1.16.0
    - thop              0.1.1.post2209072238  (⭐)
    - torch             1.7.1+cu101 (⭐)
    - torchaudio        0.7.2
    - torchvision       0.8.2+cu101
    - tqdm              4.65.0  (⭐)
    - typing_extensions 4.5.0
    - wheel             0.38.4
    
- 复现就是从main_softnode2_fixed.py文件开始的。但是使用的数据是softnode3.
- 实用小工具：
  - 在线文本比较工具：https://www.jq22.com/textDifference.
    
- resources文件夹下需要注意的事情：(以RGB1数据集为例)
    - 3mdad-rgb1_0.9_cm.npy是模型准确率为0.9时的混淆矩阵数据。
    - 3mdad-rgb1_0.9_data.npy不明白是什么意思，可以通过当前文件夹中的查看npy文件.py查看数据的shape和内容。
    - 3mdad-rgb1_best_acc.pkl **该文件中的数据不代表模型最好时得到的输出结果，遗憾的是，在processor.py的start函数中，该文件被生成到sources文件夹下，但是该文件本应该存储模型最好性能时的数据。**
    - ensemble_msegcn.py 该文件时做多视角融合的。但是它依赖的是以best_acc.pkl结尾的文件，因此目前该文件得不到91.25%的结果。
    
- scripts文件夹下需要注意的事情：
    - lingshen.sh 表示投稿之前，模型取得各个数据集上最好准确率时的配置信息。
    - lingshen_vs.sh 表示投稿之前，模型最好配置时，用于执行CAM可视化的命令行代码。
      事实上，它的内容与lingshen.sh基本一致，而lingshen_vs.sh能够执行可视化任务的原因是 main_softnode2_fixed_vs.py 比 main_softnode2_fixed.py 多了以下设置：
      ```python
        args.generate_data = False
        args.extract = True
        args.visualize = True
      ```
    - 如果要再次开启，希望你知道相关数据是如何保存的。—— 但是似乎生成vs文件是另外一个项目EfficientGCNv1_d_tc_vs。
    
- src文件下需要注意的事情：
    - dataset文件夹下需要注意的事情：
        - 各文件中与本论文所用数据集无关的代码基本全部删除。
    - model文件夹下需要注意的事情：
        - init.py 中rescale_block函数将参数中的block_args参数转换为模型真正使用的版本。
          但是这一点对于nets2和nets4不起作用，因为为了验证一审中提出的模型设计范式问题，我们图方便直接将想要的block_args作为类属性放置在EfficientGCN类中。
          另外，如果model_type中包含_Y(X)_，则模型将调用nets2345中的一个。
          其中，需要注意的是，round(0.5)=0,round(1.5)=2.round(2.5)=2,round(3.5)=4 很神奇。python3.7官网的解释是：如果距离两边一样远，会保留到偶数的一边。
        - layers_temporal.py 中为了适应nets3和4，基于Temporal_Mta2Wrapper_Layer提出了Temporal_YISHEN_Mta2Wrapper_Layer以增加输出通道数这一参数，
          同时在该函数中将模块的残差连接从原来的identity修改为Conv+BN，以实现输入通道数向输出通道数的变化。
        - nets.py 是最原始的版本，虽然有所改动，但不设计模型变动。
        - nets2.py(RGB2:84.38%) 是I(STB)+M(SB) model (a)，实现方式，仅仅是取消了原始的block_args，而使用了类内定义的属性block_args_2[1,1,0,0]。
        - nets3.py(RGB2:83.75%) 是I(TB)+M(STB) model (b)，实现方式是，修改了EfficientGCN_Blocks类内的代码，同时引入了Temporal_YISHEN_Mta2Wrapper_Layer。
        - nets4.py(RGB2:81.25%) 是I(STB)+M(TB) model (c)，实现方式是，在nets3.py的基础上，使用了block_args_2。
        - nets5.py(RGB2:82.50%) 是I(STB)+M(STB) model (d)，实现方式是，在nets.py的基础上，引入了block_args_2[1,1,1,1].
        
          
      