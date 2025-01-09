import os, yaml, argparse
from time import sleep
import sys

sys.path.append("/home/bullet/PycharmProjects/LowlightRecognition/EfficientGCNv1")
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer
import logging


def main():
    # TODO: 开始之前必须确保指定  datasetCode  granularity  configCode； 开启savingmodel；以及np.save

    parser = init_parser()
    args = parser.parse_args()
    framenum = 90
    split = int(str(args.datasetCode).split("-")[1])
    print("当前处理的数据集：{}".format(args.datasetCode))
    print("当前处理的Split：{}".format(split))
    print("当前处理的FrameNum：{}".format(framenum))
    configyaml = "./config0316_softnode3_{}/train_driveract_{}_{}.yaml".format(args.granularity, split, framenum)

    args = update_parameters(parser, args, configyaml)  # cmd > yaml > default
    work_dir = args.work_dir
    work_dir = "{}0527/{}_{}_{}".format(work_dir, args.granularity,args.datasetCode, args.configCode)
    args.work_dir = work_dir

    root_folder = args.dataset_args[args.dataset]["root_folder"]
    args.dataset_args[args.dataset]["root_folder"] = \
        str(root_folder).replace("driveract/{}".format(split),
                                 "driveract/{}level/B_f{}_{}_{}".format(args.granularity,framenum, 0, split))

    print("数据集的路径：{}".format(args.dataset_args[args.dataset]["root_folder"]))
    print("当前实验配置的种子：{}".format(args.configCode))

    if args.configCode == 1478:  # split 0
        args.seed = 1478
        args.optimizer_args["SGD"]["lr"] = 0.05
        args.optimizer_args["SGD"]["weight_decay"] = 0.0002
        args.scheduler_args["step"]["warm_up"] = 0
        args.scheduler_args["step"]["step_lr"] = [30, 50]
        args.scheduler_args["step"]["max_epoch"] = 50
        args.dataset_args[args.dataset]["train_batch_size"] = 16

    elif args.configCode == 869:  # split 1
        args.seed = 869
        args.optimizer_args["SGD"]["lr"] = 0.1
        args.optimizer_args["SGD"]["weight_decay"] = 0.0004
        args.scheduler_args["step"]["warm_up"] = 5
        args.scheduler_args["step"]["step_lr"] = [20, 50]
        args.dataset_args[args.dataset]["train_batch_size"] = 8

    elif args.configCode == 7399:  # split 2
        args.seed = 7399
        args.optimizer_args["SGD"]["lr"] = 0.05
        args.optimizer_args["SGD"]["weight_decay"] = 0.0001
        args.scheduler_args["step"]["warm_up"] = 10
        args.scheduler_args["step"]["step_lr"] = [45, 55, 65]
        args.dataset_args[args.dataset]["train_batch_size"] = 16

    elif args.configCode == 1031:  # split 2
        args.seed = 5374
        args.optimizer_args["SGD"]["lr"] = 0.05
        args.optimizer_args["SGD"]["weight_decay"] = 0.0002
        args.scheduler_args["step"]["warm_up"] = 0
        args.scheduler_args["step"]["step_lr"] = [45, 55]
        args.dataset_args[args.dataset]["train_batch_size"] = 16

    elif args.configCode == 7424:  # split 2
        args.seed = 7424
        args.optimizer_args["SGD"]["lr"] = 0.005
        args.optimizer_args["SGD"]["weight_decay"] = 0.0004
        args.scheduler_args["step"]["warm_up"] = 0
        args.scheduler_args["step"]["step_lr"] = [20, 50]
        args.dataset_args[args.dataset]["train_batch_size"] = 8

    elif args.configCode == 2962:  # split 2
        args.seed = 2962
        args.optimizer_args["SGD"]["lr"] = 0.5
        args.optimizer_args["SGD"]["weight_decay"] = 0.0005
        args.scheduler_args["step"]["warm_up"] = 0
        args.scheduler_args["step"]["max_epoch"] = 50
        args.scheduler_args["step"]["step_lr"] = [45, 55, 65]
        args.dataset_args[args.dataset]["train_batch_size"] = 64

    elif args.configCode == 144:  # split 2
        args.seed = 144
        args.optimizer_args["SGD"]["lr"] = 0.01
        args.optimizer_args["SGD"]["weight_decay"] = 0.0005
        args.scheduler_args["step"]["warm_up"] = 10
        args.scheduler_args["step"]["max_epoch"] = 50
        args.scheduler_args["step"]["step_lr"] = [20, 50]
        args.dataset_args[args.dataset]["train_batch_size"] = 8

    elif args.configCode == 5785:  # split 2
        args.seed = 5785
        args.optimizer_args["SGD"]["lr"] = 0.005
        args.optimizer_args["SGD"]["weight_decay"] = 0.0001
        args.scheduler_args["step"]["warm_up"] = 10
        args.scheduler_args["step"]["max_epoch"] = 50
        args.scheduler_args["step"]["step_lr"] = [45, 55, 65]
        args.dataset_args[args.dataset]["train_batch_size"] = 8

    elif args.configCode == 6774:  # split 2
        args.seed = 6774
        args.optimizer_args["SGD"]["lr"] = 0.1
        args.optimizer_args["SGD"]["weight_decay"] = 0.0002
        args.scheduler_args["step"]["warm_up"] = 5
        args.scheduler_args["step"]["max_epoch"] = 50
        args.scheduler_args["step"]["step_lr"] = [45, 55, 65]
        args.dataset_args[args.dataset]["train_batch_size"] = 32

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # Waiting to run
    sleep(args.delay_hours * 3600)

    # Processing
    if args.generate_data:
        g = Generator(args)
        g.start()

    elif args.extract or args.visualize:
        if args.extract:
            p = Processor(args)
            p.extract()
        if args.visualize:
            v = Visualizer(args)
            v.start()

    else:
        # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        # DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        # logging.basicConfig(filename='./tunefiles/files0527_softnode3/txt0525_{}_{}_{}_{}_{}_{}.log'.format(
        #     args.datasetCode, args.configCode, args.suffix, args.Ad, args.withself, args.gcmh), level=logging.INFO,
        #     format=LOG_FORMAT,
        #     datefmt=DATE_FORMAT,
        #     filemode="w")
        print("((((((((((((((((((((((((------------------))))))))))))))))))))))))")
        print("0527: 旨在重新生成最优参数文件，修复Saving model的代码位置！参数文件将放置在 resource0527")
        p = Processor(args)
        p.test_start()


def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config')
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--delay_hours', '-dh', type=float, default=0, help='Delay to run')

    # Processing
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualize', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')
    parser.add_argument('--suffix', '-suffix', type=str, default=None, help='Select dataset edition')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Args for creating model')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--configCode', type=int, default=144, help='con1figCode')
    parser.add_argument('--datasetCode', type=str, default="driveract-0", help='datasetCode')

    parser.add_argument('--Ad', '-ad', default=True, action='store_true', help='Ad')
    parser.add_argument('--withself', '-ws', default=True, action='store_true', help='withself')
    parser.add_argument('--gcmh', '-gcmh', type=int, default=3, help='gcmh')

    # TC Layer
    parser.add_argument('--tcr', '-tcr', type=int, default=4, help='tcr')
    parser.add_argument('--tcl', '-tcl', type=str, default="Mta2Wrapper-ST2LiteMBConv", help='tcl')
    parser.add_argument('--granularity', '-granularity', type=str, default="object", help='tcr')
    # all- object- location- action ; mid-;task-

    return parser


def update_parameters(parser, args, configyaml):
    if os.path.exists('{}'.format(configyaml)):
        with open('{}'.format(configyaml), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(configyaml))
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
