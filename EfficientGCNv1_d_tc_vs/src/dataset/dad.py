import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

from .. import utils as U


class DAD_Feeder(Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        #  dataset_path, inputs, num_frame, connect_joint, debug
        if phase == "eval":
            phase = "val"
        self.conn = connect_joint
        self.T = num_frame
        self.inputs = inputs
        if kwargs["suffix"] != None:
            logging.info("===> Dataset Edition: {}".format(kwargs["suffix"]))
            data_path = '{}/{}_data_joint_{}.npy'.format(dataset_path, phase, kwargs["suffix"])
        else:
            data_path = '{}/{}_data_joint.npy'.format(dataset_path, phase)

        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        print("*** datapath:{}".format(data_path))

        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = np.load(data_path, mmap_mode='r')
            # print("data:{}".format(np.sum(self.data)))
            assert self.data.shape[1] == 3 and self.data.shape[-1] == 1 and self.data.shape[-2] == 12
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

            if "driveract" in dataset_path:
                self.label = [int(label) - 1 for label in self.label]
            else:
                self.label = [int(label) for label in self.label]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.sample_name = self.sample_name[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.sample_name[idx]
        # seq_len = self.seq_len[idx]

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = self.multi_input(data[:, :self.T, :, :])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M))
        velocity = np.zeros((C * 2, T, V, M))
        bone = np.zeros((C * 2, T, V, M))
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
            velocity[C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        return joint, velocity, bone


class M3DADRGB1_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 3, self.T, 25, self.M))
        basePath = r"N:\3MDAD\Day\RGB1_pose_220104\skeletons"
        print("BasePath: {}.".format(basePath))

        notbinlst = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
        for i, name in enumerate(names):
            with open(os.path.join(basePath, name), 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        joint_num = int(fr.readline())
                        for joint in enumerate(range(joint_num)):
                            v = fr.readline().split(' ')[:3]
                            v = [float(v0) for v0 in v]
                            if person < self.M:
                                location[i, :, frame, joint, person] = v
        location = location[:, :, :, notbinlst, :]
        return location


class M3DADRGB2_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 3, self.T, self.V, self.M))
        # basePath = "/home/bullet/PycharmProjects/datasets/3MDAD/RGB2_pose_220104/skeletons_softnode3"
        basePath = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_skeleton"
        print("BasePath: {}.".format(basePath))

        for i, name in enumerate(names):
            with open(os.path.join(basePath, name), 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        # print(fr.readline())
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')[:3]
                            v = [float(v0) for v0 in v]
                            if joint < self.V and person < self.M:
                                location[i, :, frame, joint, person] = v
        return location


class EBDDRGB2_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 3, self.T, 25, self.M))
        notbinlst = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]

        basePath = r"N:\clip225\clip225_pose_220104\skeletons"
        print("BasePath: {}.".format(basePath))

        for i, name in enumerate(names):
            with open(os.path.join(basePath, name), 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        # print(fr.readline())
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')[:3]
                            v = [float(v0) for v0 in v]
                            if person < self.M:
                                location[i, :, frame, joint, person] = v
        location = location[:, :, :, notbinlst, :]
        return location


class DriveAct_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 3, self.T, 26, self.M))
        notbinlst = [1, 2, 3, 7, 9, 14, 17, 18, 19, 21, 23, 24]
        notbinlst = [x - 1 for x in notbinlst]
        refinelst = [1, 4, 10, 2, 3, 9, 8, 5, 6, 11, 7, 12]
        refinelst = [x - 1 for x in refinelst]

        basePath = r"P:\bullet\LowlightRecognition\coldnight\n3_storage\data_advanced_splitrgb1_0316_softnode3\driveract\midlevel\B_f90_0_0\val_data_joint.npy"
        print("BasePath: {}.".format(basePath))

        # for i, name in enumerate(names):
        #     with open(os.path.join(basePath, name), 'r') as fr:
        #         frame_num = int(fr.readline())
        #         for frame in range(frame_num):
        #             if frame >= self.T:
        #                 break
        #             person_num = int(fr.readline())
        #             for person in range(person_num):
        #                 # print(fr.readline())
        #                 joint_num = int(fr.readline())
        #                 for joint in range(joint_num):
        #                     v = fr.readline().split(' ')[:3]
        #                     v = [float(v0) for v0 in v]
        #                     if person < self.M:
        #                         location[i, :, frame, joint, person] = v
        # location = location[:, :, :, notbinlst, :]
        # location = location[:, :, :, refinelst, :]
        location = np.load(basePath)
        assert location.shape[-1] == 1 and location.shape[-2] == 12 and location.shape[-3] == 90
        return location
