import numpy as np
import pickle
import os, re
from tqdm import tqdm

# 第一步得到各个序列的50个ID值。
# 验证本机和服务器上得到的50个ID值是否一致。

# 第二步将50章图片copy到其他的文件夹中。

out_imgs_folder_path = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_imgs"
if os.path.exists(out_imgs_folder_path):
    import shutil

    shutil.rmtree(out_imgs_folder_path)
os.mkdir(out_imgs_folder_path)

imgs_folder_path = r"N:\3MDAD\Day\RGB2_mix"
imgs_folders = os.listdir(imgs_folder_path)
imgs_folders.sort(key=lambda x: [int(re.findall(r"\d+", x)[0]),
                                 int(re.findall(r"\d+", x)[1])])

val_index_data_path = r"P:\bullet\LowlightRecognition\coldnight\n2\Skeletons预处理\val_outindex.npy.npz"
val_index_data = np.load(val_index_data_path)
val_index = val_index_data["outArr"]
val_sample_name = val_index_data["sample_name"]
val_sample_name = [sample_name.split(".")[0] for sample_name in val_sample_name]

print(val_index)
print(val_sample_name)

val_sample_path_lst = [os.path.join(imgs_folder_path, sample_name) for sample_name in val_sample_name]
out_sample_path_lst = [os.path.join(out_imgs_folder_path, sample_name) for sample_name in val_sample_name]

for sample_index, val_sample_path in enumerate(val_sample_path_lst):
    print(sample_index)
    imgs_index_lst = val_index[sample_index]
    imgs_index_lst = [int(index) for index in imgs_index_lst]
    # print(imgs_index_lst)

    out_sample_path = out_sample_path_lst[sample_index]
    if not os.path.exists(out_sample_path):
        os.mkdir(out_sample_path)

    imgs_name_lst = os.listdir(val_sample_path)
    imgs_name_lst.sort(key=lambda x: int(x.split(".")[0][(str(x).index("F")) + 1:]))
    selected_imgs_name_lst = np.array(imgs_name_lst)[imgs_index_lst]

    imgs_path_lst = [os.path.join(val_sample_path, img_name) for img_name in selected_imgs_name_lst]
    out_imgs_path_lst = [os.path.join(out_sample_path, img_name) for img_name in selected_imgs_name_lst]

    for i in tqdm(range(len(imgs_path_lst))):
        img_path = imgs_path_lst[i]
        out_img_path = out_imgs_path_lst[i]
        shutil.copyfile(img_path, out_img_path)


