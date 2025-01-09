import os
import numpy as np
import pickle

# 面向第二个阶段的实验Step
# 2023年04月16日的解释。
# 特别地，之所以进行第二阶段的实验，是因为基于余弦的结果中，object的val值和location的val值结果不好。
# 而最后探究的结果，中obejct的split0和split1的随机种子得到改善。同时location的split2的随机种子也得到了改善。
#

basePath = "/home/meng/PycharmProjects/EfficientSeries/EfficientGCNv1_d_tc_driveact/resource0527"

# split0_path = os.path.join(basePath, "valid0526_location_driveract-0_5785_data.npy")
# split0_path = os.path.join(basePath, "valid0526_location_driveract-0_8642_data.npy")
# split1_path = os.path.join(basePath, "valid0526_location_driveract-1_1478_data.npy")
# split2_path = os.path.join(basePath, "valid0526_location_driveract-2_6774_data.npy")

# 66.29032037799651 70.60626587677845 57.40440706196359 61.505022113809225

# split0_path = os.path.join(basePath, "test0526_location_driveract-0_5785_data.npy")
# split1_path = os.path.join(basePath, "test0526_location_driveract-1_1478_data.npy")
# split2_path = os.path.join(basePath, "test0526_location_driveract-2_6774_data.npy")
# 55.5161660606165 55.71719089567796 60.336839914326234 56.54964995464624
# ------------------------
# split0_path = os.path.join(basePath, "valid0526_location_driveract-0_7470_data.npy")
# split1_path = os.path.join(basePath, "valid0526_location_driveract-1_144_data.npy")
# split2_path = os.path.join(basePath, "valid0526_location_driveract-2_6774_data.npy")
# 67.04373231440331 69.19651898695194 57.40440706196359 61.71924551052813 (⭐)

# split0_path = os.path.join(basePath, "test0526_location_driveract-0_7470_data.npy")
# split1_path = os.path.join(basePath, "test0526_location_driveract-1_144_data.npy")
# split2_path = os.path.join(basePath, "test0526_location_driveract-2_6774_data.npy")
# 55.5161660606165 55.71719089567796 60.336839914326234 56.54964995464624 (⭐)



split0_path = os.path.join(basePath, "valid0526_object_driveract-0_144_data.npy")
split1_path = os.path.join(basePath, "valid0526_object_driveract-1_144_data.npy")
split2_path = os.path.join(basePath, "valid0526_object_driveract-2_4059_data.npy")
# 70.06399904059727 56.92715984499602 55.95802529882721 57.883782453003185
# 70.06399904059727 56.92715984499602 55.62781087294128 57.84198980144436  (⭐)

# split0_path = os.path.join(basePath, "test0526_object_driveract-0_144_data.npy")
# split1_path = os.path.join(basePath, "test0526_object_driveract-1_144_data.npy")
# split2_path = os.path.join(basePath, "test0526_object_driveract-2_4059_data.npy")
# 56.14536528317902 52.60028056501345 40.74088317766158 46.124470192826955
# 56.14536528317902 52.60028056501345 42.08052055218951 46.977794093155325  (⭐)


data0 = np.load(split0_path)
split0_label, split0_target = data0[0], data0[1]

data1 = np.load(split1_path)
split1_label, split1_target = data1[0], data1[1]

data2 = np.load(split2_path)
split2_label, split2_target = data2[0], data2[1]

combine_label, combine_target = [], []

combine_label.extend(list(split0_label))
combine_label.extend(list(split1_label))
combine_label.extend(list(split2_label))

combine_target.extend(list(split0_target))
combine_target.extend(list(split1_target))
combine_target.extend(list(split2_target))

assert len(combine_target) == len(combine_target) and len(split0_target) != len(split1_target)

from sklearn.metrics import balanced_accuracy_score

balanced_acc_0 = balanced_accuracy_score(split0_label, split0_target) * 100
balanced_acc_1 = balanced_accuracy_score(split1_label, split1_target) * 100
balanced_acc_2 = balanced_accuracy_score(split2_label, split2_target) * 100
all_balanced_acc = balanced_accuracy_score(combine_label, combine_target) * 100

print(balanced_acc_0, balanced_acc_1, balanced_acc_2, all_balanced_acc)
