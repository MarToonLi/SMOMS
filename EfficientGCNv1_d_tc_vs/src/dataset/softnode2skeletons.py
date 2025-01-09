import numpy as np
import pickle
import os
from tqdm import tqdm

"""
1 将softnode3的npy文件转换成skeleton文件的格式；文件的命名需要结合label来做
1.0 需要将所有的py文件执行都切换到本机上！
1.1 需要成功读取到data和label的文件，对应上文件名；
1.2 总结skeleton的文件格式。
1.3 
2 当上一步完成后，需要做另一件事情：将所有测试样本（160，起码四分之一）的CAM图保存下来。
"""


def gen_data_joints(line, skeletons_folder, images_folder_name):
    if not os.path.exists(skeletons_folder):
        os.makedirs(skeletons_folder)
    with open('{}/{}'.format(skeletons_folder, images_folder_name), 'a') as f:
        if type(line) is not list:
            f.write(str(line))
        else:
            for i in line:
                f.write(str(i))
                f.write(" ")
        f.write("\n")


def sample2skeleton(sample, softnode3_skeleton_folder, sample_name):
    assert sample.shape == (3, 50, 12, 1)
    C, T, V, M = sample.shape
    sample = np.transpose(sample, (1, 2, 0, 3))
    sample = np.squeeze(sample)
    assert sample.shape == (50, 12, 3)
    gen_data_joints(50, softnode3_skeleton_folder, sample_name)
    for t in range(T):
        gen_data_joints(1, softnode3_skeleton_folder, sample_name)
        gen_data_joints(12, softnode3_skeleton_folder, sample_name)
        for v in range(V):
            node = list(sample[t, v, :])
            gen_data_joints(node, softnode3_skeleton_folder, sample_name)


softnode3_skeleton_folder = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_skeleton"
softnode3_folder = r"P:\bullet\LowlightRecognition\coldnight\n3_storage\data_advanced_splitrgb1_0316_softnode3\3mdad\clip50\RGB2"
joint_num = 25
person_num = 1

val_data_path = os.path.join(softnode3_folder, "val_data_joint.npy")
val_label_path = os.path.join(softnode3_folder, "val_label.pkl")

val_data = np.load(val_data_path, mmap_mode='r')

with open(val_label_path, 'rb') as f:
    sample_name_lst, label_lst = pickle.load(f, encoding='latin1')
    label_lst = [int(label) for label in label_lst]

print(len(val_data))
print(len(sample_name_lst))
print(val_data.shape)

if os.path.exists(softnode3_skeleton_folder):
    import shutil

    shutil.rmtree(softnode3_skeleton_folder)

os.mkdir(softnode3_skeleton_folder)

# sample2skeleton(sample, softnode3_skeleton_folder, sample_name)

for sample_index in tqdm(range(len(val_data))):
    sample = val_data[sample_index]
    sample_name = sample_name_lst[sample_index]
    sample2skeleton(sample, softnode3_skeleton_folder, sample_name)
