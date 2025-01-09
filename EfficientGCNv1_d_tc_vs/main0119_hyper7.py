import os, yaml, argparse
from time import sleep
import sys

from src.generator import Generator
from src.processor import Processor


def main(args):
    sleep(args.delay_hours * 3600)

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


def init_seed(seed=1):
    import torch
    import numpy as np
    import random
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def spot(arg, config, checkpoint_dir=None):
    import copy

    arg = copy.deepcopy(arg)

    arg.seed = config["seed"]
    arg.optimizer_args["SGD"]["lr"] = config["lr"]
    arg.dataset_args[arg.dataset]["train_batch_size"] = config["bs"]
    arg.optimizer_args["SGD"]["weight_decay"] = config["wd"]
    arg.scheduler_args["cosine"]["max_epoch"] = config["max_epoch"]
    arg.scheduler_args["cosine"]["warm_up"] = config["warm"]

    init_seed(arg.seed)
    main(arg)


if __name__ == '__main__':
    import time

    start_time = time.time()

    # dataset, point, part = "3mdad", "RGB1", "joint"
    # dataset, point, part = "3mdad", "RGB2", "joint"
    # dataset, part = "dad", "joint"
    # dataset, part = "ebdd",  "joint"
    # dataset, part = "driveract", "0"
    dataset, part = "driveract", "1"

    if dataset == "3mdad":
        configyaml = "./config0316_softnode3/train_{}_{}.yaml".format(
            dataset, point
        )
        df_name = "{}_{}_{}".format(dataset, point, part)

    elif dataset == "dad":
        configyaml = "./config0316_softnode3/train_{}.yaml".format(
            dataset
        )
        df_name = "{}_{}".format(dataset, part)

    elif dataset == "ebdd":
        configyaml = "./config0316_softnode3/train_{}.yaml".format(
            dataset
        )
        df_name = "{}_{}".format(dataset, part)
    elif dataset == "driveract":
        configyaml = "./config0316_softnode3/train_{}_{}.yaml".format(
            dataset, part
        )
        df_name = "{}_{}".format(dataset, part)
    else:
        raise ValueError("error dataset!")
    from loguru import logger

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = init_parser()
    args = parser.parse_args()
    arg = update_parameters(parser, args, configyaml)  # cmd > yaml > default

    import random
    import ray
    import numpy as np
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.logger import CSVLoggerCallback

    random.seed(1)
    np.random.seed(1)
    ray.shutdown()

    # gpus = "0,1,2,3,4"
    # gpus = "0,1,2"
    gpus = "1,2,3,4,5"

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_gpus = len(gpus.split(","))
    ray.init(num_cpus=num_gpus * 12, num_gpus=num_gpus)
    # ray.init()

    config = {
        "seed": tune.randint(0, 10000),
        "lr": tune.choice([0.1, 0.01, 0.001, 0.0001, 0.5, 0.05, 0.005]),
        "bs": tune.choice([16, 32, 64]),
        "wd": tune.choice([0.0001, 0.0002, 0.0003, 0.0004, 0.0005]),
        "max_epoch": tune.choice([50, 60, 70, 80]),
        "warm": tune.choice([0, 5, 10]),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=80,
        grace_period=4,
        reduction_factor=4)

    reporter = CLIReporter(metric_columns=["accuracy", "training_iteration"])

    from functools import partial

    analysis = tune.run(
        partial(spot, arg),
        name=df_name,
        num_samples=80,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        callbacks=[CSVLoggerCallback()],
        verbose=2,
        local_dir="/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/ray_results",

    )
    import pandas as pd

    # Ad  withself  gcmh

    pd.set_option('display.max_columns', None)
    df = analysis.dataframe()
    df.to_csv(
        "./tunefiles/files0422/T0423_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(df_name, arg.model_type.split("-")[1],
                                                                         arg.suffix, arg.Ad, arg.withself,
                                                                         arg.gcmh, arg.tcl, arg.tcr))

    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    logger.info("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
    logger.info("====" * 20)

    ray.shutdown()

    # import copy
    #
    # arg = copy.deepcopy(arg)
    # arg.seed = best_trial.config["seed"]
    # arg.optimizer_args["SGD"]["lr"] = best_trial.config["lr"]
    # arg.dataset_args[arg.dataset]["train_batch_size"] = best_trial.config["bs"]
    # arg.optimizer_args["SGD"]["weight_decay"] = best_trial.config["wd"]
    # arg.scheduler_args["cosine"]["max_epoch"] = best_trial.config["max_epoch"]
    # arg.scheduler_args["cosine"]["warm_up"] = best_trial.config["warm"]
    #
    # init_seed(arg.seed)
    # main(arg)

    # spend_time = time.time() - start_time
    # logger.info("spend_time:{}".format(spend_time))
