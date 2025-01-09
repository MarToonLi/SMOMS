import os, yaml, argparse
from time import sleep
import sys

sys.path.append("/home/bullet/PycharmProjects/LowlightRecognition/EfficientGCNv1")
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer


def main():
    # Loading parameters
    parser = init_parser()
    args = parser.parse_args()

    if args.datasetCode == "3mdad-rgb1":
        configyaml = "./config0316_softnode3/train_3mdad_RGB1.yaml"
    elif args.datasetCode == "3mdad-rgb2":
        configyaml = "./config0316_softnode3/train_3mdad_RGB2.yaml"
    elif args.datasetCode == "dad":
        configyaml = "./config0316_softnode3/train_dad.yaml"
    elif args.datasetCode == "ebdd":
        configyaml = "./config0316_softnode3/train_ebdd.yaml"
    else:
        print(args.datasetCode)
        raise ValueError("error datasetCode!")

    args = update_parameters(parser, args, configyaml)  # cmd > yaml > default

    if args.configCode == 2561:
        args.seed = 2561
        # 0	50	16	0.1	50	2561	0	0.0002
        args.optimizer_args["SGD"]["lr"] = 0.1
        args.optimizer_args["SGD"]["weight_decay"] = 0.0002
        args.scheduler_args["cosine"]["max_epoch"] = 50
        args.scheduler_args["cosine"]["warm_up"] = 0
    elif args.configCode == 7337:
        args.seed = 7337
        # 0	50	16	0.1	50	7337	5	0.0004
        args.optimizer_args["SGD"]["lr"] = 0.1
        args.optimizer_args["SGD"]["weight_decay"] = 0.0004
        args.scheduler_args["cosine"]["max_epoch"] = 50
        args.scheduler_args["cosine"]["warm_up"] = 5
    elif args.configCode == 7456:
        args.seed = 7456
        # 0	80	16	0.05	80	7456	0	0.0003
        args.optimizer_args["SGD"]["lr"] = 0.05
        args.optimizer_args["SGD"]["weight_decay"] = 0.0003
        args.scheduler_args["cosine"]["max_epoch"] = 80
        args.scheduler_args["cosine"]["warm_up"] = 0
    elif args.configCode == 9987:
        args.seed = 9987
        # 0	60	16	0.1	60	9987	10	0.0005
        args.optimizer_args["SGD"]["lr"] = 0.1
        args.optimizer_args["SGD"]["weight_decay"] = 0.0005
        args.scheduler_args["cosine"]["max_epoch"] = 60
        args.scheduler_args["cosine"]["warm_up"] = 10
    elif args.configCode == 9552:
        args.seed = 9552
        # 0	60	16	0.1	60	9987	10	0.0005
        args.optimizer_args["SGD"]["lr"] = 0.05
        args.optimizer_args["SGD"]["weight_decay"] = 0.0003
        args.scheduler_args["cosine"]["max_epoch"] = 80
        args.scheduler_args["cosine"]["warm_up"] = 0
        args.dataset_args["3mdad_downsampled"]["train_batch_size"] = 32
        args.dataset_args["3mdad_downsampled"]["train_batch_size"] = 32
    elif args.configCode == 144:
        args.seed = 144
        # 0	60	16	0.1	60	9987	10	0.0005
        args.optimizer_args["SGD"]["lr"] = 0.01
        args.optimizer_args["SGD"]["weight_decay"] = 0.0005
        args.scheduler_args["cosine"]["max_epoch"] = 70
        args.scheduler_args["cosine"]["warm_up"] = 5

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
        import logging
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        logging.basicConfig(filename='./tunefiles/files0415_softnode3_main/txt0424_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
            args.datasetCode, args.configCode, args.suffix, args.Ad, args.withself, args.gcmh, args.tcl, args.tcr
        ), level=logging.INFO,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            filemode="w")
        p = Processor(args)
        p.start()


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

    parser.add_argument('--device', type=int, default=5, help='device')
    parser.add_argument('--configCode', type=int, default=144, help='configCode')
    parser.add_argument('--datasetCode', type=str, default="ebdd", help='datasetCode')

    parser.add_argument('--Ad', '-ad', default=True, action='store_true', help='Ad')
    parser.add_argument('--withself', '-ws', default=True, action='store_true', help='withself')
    parser.add_argument('--gcmh', '-gcmh', type=int, default=3, help='gcmh')

    # TC Layer
    parser.add_argument('--tcr', '-tcr', type=int, default=4, help='tcr')
    parser.add_argument('--tcl', '-tcl', type=str, default="Mta2Wrapper-ST2LiteMBConv", help='tcl')

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
