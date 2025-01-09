import os, pickle, logging, numpy as np
from torch.utils.data import Dataset


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
