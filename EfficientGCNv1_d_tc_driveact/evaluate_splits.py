import os
import numpy as np
import pickle

basePath = "/home/meng/PycharmProjects/EfficientSeries/EfficientGCNv1_d_tc_driveact/resources"

# split0_path = os.path.join(basePath, "valid_driveract-0_1150_data.npy")
# split1_path = os.path.join(basePath, "valid_driveract-1_7337_data.npy")
# split2_path = os.path.join(basePath, "valid_driveract-2_2561_data.npy")
# # 68.1431443805334 58.865559040301285 62.60520800584817 60.41331299607759
# # 68.1431443805334 58.445380294755566 62.60520800584817 60.26243366926192

# split0_path = os.path.join(basePath, "valid0526_action_driveract-0_5373_data.npy")
# split1_path = os.path.join(basePath, "valid0526_action_driveract-1_7470_data.npy")
# split2_path = os.path.join(basePath, "valid0526_action_driveract-2_7456_data.npy")
# 70.92378043272394 64.75592022454359 60.37760718449272 65.12396698823872

# split0_path = os.path.join(basePath, "test0526_action_driveract-0_5373_data.npy")
# split1_path = os.path.join(basePath, "test0526_action_driveract-1_7470_data.npy")
# split2_path = os.path.join(basePath, "test0526_action_driveract-2_7456_data.npy")
# 61.89049126636201 55.47446654322845 49.09737808827019 53.36087281917319

# split0_path = os.path.join(basePath, "valid0526_all_driveract-0_7337_data.npy")
# split1_path = os.path.join(basePath, "valid0526_all_driveract-1_5374_data.npy")
# split2_path = os.path.join(basePath, "valid0526_all_driveract-2_4059_data.npy")
# 19.06673253626642 19.737313868401525 13.624850097216287 12.791319158433037

# split0_path = os.path.join(basePath, "test0526_all_driveract-0_7337_data.npy")
# split1_path = os.path.join(basePath, "test0526_all_driveract-1_5374_data.npy")
# split2_path = os.path.join(basePath, "test0526_all_driveract-2_4059_data.npy")
# 14.715558749874727 10.43252959484017 13.390692181269237 9.673680185045564

# split0_path = os.path.join(basePath, "valid0526_task_driveract-0_7470_data.npy")
# S_valid0526_task_driveract-0_3082_data
# split0_path = os.path.join(basePath, "S_valid0526_task_driveract-0_3082_data.npy")
# split1_path = os.path.join(basePath, "valid0526_task_driveract-1_7456_data.npy")
# split2_path = os.path.join(basePath, "valid0526_task_driveract-2_7456_data.npy")
# 41.75444186967266 43.30140316123083 42.374067663921096 41.411646935175206
# 44.2940530248159 43.30140316123083 42.374067663921096 42.096991809851076

# split0_path = os.path.join(basePath, "test0526_task_driveract-0_3082_data.npy")
# split0_path = os.path.join(basePath, "test0526_task_driveract-0_7470_data.npy")
# split1_path = os.path.join(basePath, "test0526_task_driveract-1_7456_data.npy")
# split2_path = os.path.join(basePath, "test0526_task_driveract-2_7456_data.npy")
# split2_path = os.path.join(basePath, "test0526_task_driveract-2_144_data.npy")
# split2_path = os.path.join(basePath, "test0526_task_driveract-2_5373_data.npy")

# 29.485259780784755 34.81330793424455 28.19430024657759 31.146089221001834   7470
# 36.94627761260592 34.81330793424455 28.19430024657759 33.31032069171279  3082
# 36.94627761260592 34.81330793424455 28.95964824724389 33.44019784069995  144
# 36.94627761260592 34.81330793424455 30.367086356817836 33.77893999895086  5373

# split0_path = os.path.join(basePath, "valid0526_object_driveract-0_8642_data.npy")
# split1_path = os.path.join(basePath, "valid0526_object_driveract-1_2561_data.npy")
# split2_path = os.path.join(basePath, "valid0526_object_driveract-2_4059_data.npy")
# split2_path = os.path.join(basePath, "S_valid0526_object_driveract-2_144_data.npy")
# 65.25558909946848 56.03857570983662 55.62781087294128 55.239981578958954
# 65.25558909946848 56.03857570983662 55.84194098129509 55.47725985128756

# split0_path = os.path.join(basePath, "test0526_object_driveract-0_8642_data.npy")
# split1_path = os.path.join(basePath, "test0526_object_driveract-1_2561_data.npy")
# split2_path = os.path.join(basePath, "test0526_object_driveract-2_4059_data.npy")
# split2_path = os.path.join(basePath, "test0526_object_driveract-2_4059_data.npy")
# 55.2902250148602 50.31725649796744 42.08052055218951 46.0886534781399

# split0_path = os.path.join(basePath, "valid0526_mid_driveract-0_1150_data.npy")
# split1_path = os.path.join(basePath, "valid0526_mid_driveract-1_1150_data.npy")
# split2_path = os.path.join(basePath, "valid0526_mid_driveract-2_2561_data.npy")
# 68.1431443805334 58.865559040301285 62.60520800584817 60.41331299607759

# split0_path = os.path.join(basePath, "test0526_mid_driveract-0_1150_data.npy")
# split1_path = os.path.join(basePath, "test0526_mid_driveract-1_1150_data.npy")
# split2_path = os.path.join(basePath, "test0526_mid_driveract-2_2561_data.npy")
# 55.45075495125968 54.94569637172637 44.84256430935385 51.387272463101716

split0_path = os.path.join(basePath, "valid0526_location_driveract-0_7470_data.npy")
split1_path = os.path.join(basePath, "valid0526_location_driveract-1_144_data.npy")
# split2_path = os.path.join(basePath, "valid0526_location_driveract-2_4059_data.npy")
split2_path = os.path.join(basePath, "S_valid0526_location_driveract-2_5373_data.npy")

# 67.04373231440331 69.19651898695194 56.04508494672397 61.77055796893596
# 67.04373231440331 69.19651898695194 54.37086196899327 59.29676824043844

# split0_path = os.path.join(basePath, "test0526_location_driveract-0_7470_data.npy")
# split1_path = os.path.join(basePath, "test0526_location_driveract-1_144_data.npy")
# split2_path = os.path.join(basePath, "test0526_location_driveract-2_7337_data.npy")
# split2_path = os.path.join(basePath, "test0526_location_driveract-2_5373_data.npy")
# split2_path = os.path.join(basePath, "test0526_location_driveract-2_4059_data.npy")
# 59.37060275717778 56.18709555235433 53.370483479718935 55.72151442199741
# 59.37060275717778 56.18709555235433 53.77885903072502 56.246453501174784  7337
# 59.37060275717778 56.18709555235433 55.10564984971812 56.53185599927749  5373


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
# 67.12536113388963 58.445380294755566 62.60520800584817 59.83548876439251