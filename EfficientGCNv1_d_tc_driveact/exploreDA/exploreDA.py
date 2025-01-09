import os
import os.path
import numpy as np
import random
import csv
import re
import cv2
import torch
import torch.utils.data as data_utl
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import json
from PIL import Image

basePath = r"/home/bullet/PycharmProjects/datasets/Driver&Act/data"
outPath = r"/home/bullet/PycharmProjects/datasets/Driver&Act/data/openpose_files"


def gen_data_joints(line, skeletons_folder, images_folder_name):
    if not os.path.exists(skeletons_folder):
        os.makedirs(skeletons_folder)
    with open('{}/{}.skeleton'.format(skeletons_folder, images_folder_name), 'a') as f:
        if type(line) is not list:
            f.write(str(line))
        else:
            for i in line:
                f.write(str(i))
                f.write(" ")
        f.write("\n")


def make_dataset(task, split, mode):
    if task not in ["midlevel", "objectlevel", "tasklevel"]:
        raise ValueError("Error task!")
    if split not in [0, 1, 2]:
        raise ValueError("Error split!")
    if mode not in ["train", "val", "test"]:
        raise ValueError("Error mode!")

    # midlevel  objectlevel  tasklevel
    taskPath = os.path.join(basePath, "{}.csv".format(task if task != "tasklevel" else "firstlevel"))
    with open(taskPath) as f:
        reader = csv.reader(f)
        label_list = [row[1] for row in reader]

    split_file_Path = os.path.join(basePath, "inner_mirror/{}.chunks_90.split_{}.{}.csv".format(task, split, mode))
    print("===> Split File: {}".format(split_file_Path))
    df_splitfile = pd.read_csv(split_file_Path)

    skeletons_folder_name = os.path.join(outPath, "{}.chunks_90.split_{}.{}.skeletons".format(task, split, mode))

    return label_list, df_splitfile, skeletons_folder_name


def label2id(label_list):
    label2id_dict = {}
    for i, label in enumerate(label_list):
        label2id_dict[label] = i
    return label2id_dict


def csv2npy_main(task, split, mode):
    label_list, df_splitfile, skeletons_folder_name = make_dataset(task, split, mode)
    label2id_dict = label2id(label_list)
    form_clip(label2id_dict, df_splitfile, skeletons_folder_name)


def npy2json(label2id_dict, df_splitfile):
    # 帅选出当前mode中的驾驶员序号---->驾驶员第几次实验的数据----> 每一个clip的数据。
    participant_id_lst = set(df_splitfile.iloc[:, 0])
    # print(participant_id_lst)
    for participant_id in participant_id_lst:  # 10 drivers
        clip_1 = df_splitfile[df_splitfile["participant_id"] == participant_id]
        file_id_lst = set(clip_1.iloc[:, 1])
        for file_id in file_id_lst:  # 1 2
            clip_2 = clip_1[clip_1["file_id"] == file_id]
            openpose_path = os.path.join(basePath, "openpose_3d", file_id + ".openpose.3d.csv")
            df = pd.read_csv(openpose_path)
            clip_info = {}
            for i in range(len(clip_2)):
                _, _, _, frame_start, frame_end, activity, chunk_id = clip_2.iloc[i, :]

                labelid = label2id_dict[activity]
                data = df.iloc[frame_start:frame_end + 1, df.columns != "timestamp"]  # 去除 timestamp
                data = data.iloc[:, 1:]  # 去除 frame_id
                data = np.array(data)  # 当前clip片段中的N帧26节点的数据信息。
                print(data.shape)
                assert len(data) == frame_end - frame_start + 1
                clip_info[i] = data.reshape((len(data), 26, 4))


def form_clip(label2id_dict, df_splitfile, skeletons_folder_name):
    """
    1 提供
    ————————————————————————————————————
    1 首先记录总共的帧数，包含缺失帧；
    2 记录当前帧的body数目：0，1，2，3 ，但是在当前的文件中，仅存在一个两个选择，0或者1。
    3 如果body数目为0，则直接进入下一帧；如果body数目为1，则输出numJoint为26.同时下一行输出为
    Returns:
    """
    # 帅选出当前mode中的驾驶员序号---->驾驶员第几次实验的数据----> 每一个clip的数据。
    participant_id_lst = set(df_splitfile.iloc[:, 0])
    # print(participant_id_lst)
    for participant_id in participant_id_lst:  # 10 drivers
        clip_1 = df_splitfile[df_splitfile["participant_id"] == participant_id]
        file_id_lst = set(clip_1.iloc[:, 1])
        for file_id in file_id_lst:  # 1 2
            clip_2 = clip_1[clip_1["file_id"] == file_id]
            openpose_path = os.path.join(basePath, "openpose_3d", file_id + ".openpose.3d.csv")
            print("================ {} -- {}  =================".format(file_id, len(clip_2)))

            df = pd.read_csv(openpose_path)
            clip_info = {}
            for i in tqdm(range(len(clip_2))):
                _, file_id, annotation_id, frame_start, frame_end, activity, chunk_id = clip_2.iloc[i, :]

                labelid = label2id_dict[activity]
                data = df.iloc[frame_start:frame_end + 1, df.columns != "timestamp"]  # 去除 timestamp
                data = data.iloc[:, 1:]  # 去除 frame_id
                data = np.array(data)  # 当前clip片段中的N帧26节点的数据信息。
                data = data.reshape((len(data), 26, 4))
                # print(data.shape)
                assert len(data) == frame_end - frame_start + 1
                clip_info[i] = data

                # 构造文件
                skeleton_file_name = "{}_{}_{}_{}_{}".format(participant_id - 1, file_id[int(file_id.index("run") + 3)],
                                                             annotation_id, labelid, chunk_id)
                # 记录总的帧数 int(len(data))
                gen_data_joints(int(len(data)), skeletons_folder_name, skeleton_file_name)

                for j in range(len(data)):
                    # 第 J 帧
                    # 判断第J帧是否为缺失帧。
                    if np.sum(data[j]) == 0:
                        gen_data_joints(0, skeletons_folder_name, skeleton_file_name)
                    else:
                        gen_data_joints(1, skeletons_folder_name, skeleton_file_name)
                        # 记录所包含的节点数目
                        gen_data_joints(26, skeletons_folder_name, skeleton_file_name)
                        for n in range(26):
                            gen_data_joints(list(data[j, n]), skeletons_folder_name, skeleton_file_name)


if __name__ == '__main__':
    """
    1 逐帧分析，如果检测不到people则，填充为0；
    2 计划：先读取csv；再得到每一帧对应的26*4=104个数值；后期通过切片或者12个节点数据。
    3 借助split文件，切分clip；N个clip----M个帧----P个节点----C个通道值（C=4）—— 得到一个Json文件。
    4 生成一个pkl文件。
    5 更改计划，改造成：skeleton的格式：每个clip一个skeleton文件；第一个
    
    """
    # csv2npy_main(task="midlevel", split=0, mode="train")
    # csv2npy_main(task="midlevel", split=0, mode="val")
    # csv2npy_main(task="midlevel", split=0, mode="test")
    # csv2npy_main(task="midlevel", split=1, mode="train")
    # csv2npy_main(task="midlevel", split=1, mode="val")
    # csv2npy_main(task="midlevel", split=1, mode="test")
    # csv2npy_main(task="midlevel", split=2, mode="train")
    # csv2npy_main(task="midlevel", split=2, mode="val")
    # csv2npy_main(task="midlevel", split=2, mode="test")

    #     if task not in ["midlevel", "objectlevel", "tasklevel"]:
    csv2npy_main(task="tasklevel", split=0, mode="train")
    csv2npy_main(task="tasklevel", split=0, mode="val")
    csv2npy_main(task="tasklevel", split=0, mode="test")
    csv2npy_main(task="tasklevel", split=1, mode="train")
    csv2npy_main(task="tasklevel", split=1, mode="val")
    csv2npy_main(task="tasklevel", split=1, mode="test")
    csv2npy_main(task="tasklevel", split=2, mode="train")
    csv2npy_main(task="tasklevel", split=2, mode="val")
    csv2npy_main(task="tasklevel", split=2, mode="test")
