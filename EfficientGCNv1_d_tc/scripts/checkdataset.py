import os

# path1 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1/3mdad/clip50/RGB1/"
# path2 = "/home/bullet/PycharmProjects/beifen/3mdad/clip50/RGB1/"
# path1 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1/ebdd/clip225/none"
# path2 = "/home/bullet/PycharmProjects/beifen/ebdd/clip225/none"
path1 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1/dad/clip225/ratio"
path2 = "/home/bullet/PycharmProjects/beifen/dad/clip225/ratio"

datasets = os.listdir(path1)
datasets2 = os.listdir(path2)
datasets_path1 = [os.path.join(path1, dataset) for dataset in datasets]
datasets_path2 = [os.path.join(path2, dataset) for dataset in datasets]
print("check1 路径下的文件数目一致？：{}-{}".format(len(datasets), len(datasets2)))
import numpy as np

for dataset in datasets:
    if "1"  in dataset:
        continue
    if "pkl" in dataset:
        continue
    dataset_path1 = os.path.join(path1, dataset)
    dataset_path2 = os.path.join(path2, dataset)
    data1 = np.load(dataset_path1)
    data2 = np.load(dataset_path2)
    if "softnode" in dataset:
        print("==> [Dataset]:{} Sum:{}_{}".format(dataset, np.sum(data1), np.sum(data2)))

    if np.sum(data1)!= np.sum(data2):
        print("==> [Dataset]:{} Sum:{}_{}".format(dataset, np.sum(data1), np.sum(data2)))
