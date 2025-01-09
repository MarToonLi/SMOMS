import logging, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import sys

from . import utils as U


class Visualizer():
    def __init__(self, args):
        self.args = args
        U.set_logging(args)
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['3mdad_downsampled1'] = ["Safe driving A1",
                                                   "Doing hair and makeup A2",
                                                   "Adjusting radio A3",
                                                   "GPS operating A4",
                                                   "Writing message using right hand A5",
                                                   "Writing message using left hand A6",
                                                   "Talking phone using right hand A7",
                                                   "Talking phone using left hand A8",
                                                   "Having picture A9",
                                                   "Talking to passenger A10",
                                                   "Singing or dancing A11",
                                                   "Fatigue and somnolence A12",
                                                   "Drinking using right hand A13",
                                                   "Drinking using left hand A14",
                                                   "Reaching behind A15",
                                                   "Smoking A16",
                                                   ]
        self.action_names['3mdad_downsampled2'] = ["Safe driving A1",
                                                   "Doing hair and makeup A2",
                                                   "Adjusting radio A3",
                                                   "GPS operating A4",
                                                   "Writing message using right hand A5",
                                                   "Writing message using left hand A6",
                                                   "Talking phone using right hand A7",
                                                   "Talking phone using left hand A8",
                                                   "Having picture A9",
                                                   "Talking to passenger A10",
                                                   "Singing or dancing A11",
                                                   "Fatigue and somnolence A12",
                                                   "Drinking using right hand A13",
                                                   "Drinking using left hand A14",
                                                   "Reaching behind A15",
                                                   "Smoking A16",
                                                   ]
        self.action_names['ebdd'] = ["A1",
                                     "A2",
                                     "A3",
                                     "A4",
                                     "A5",
                                     ]
        self.action_names['driveract'] = ["C{}".format(i) for i in range(32)]
        self.font_sizes = {
            "3mdad_downsampled1": 14,
            "3mdad_downsampled2": 14,
            "ebdd": 14,
            "driveract": 14,

        }

    def start(self):
        self.read_data()

    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction{}.npz'.format(self.args.config)
        data = np.load(data_file)
        action_type_lst = [4, 5, 7, 9, 11, 14, 15, 17, 19, 20, 22, 23, 26, 32]

        sample_num = len(data['name'])

        for i in range(sample_num):
            action_type = int(str(data['name'][i]).split("_")[-2])
            if action_type not in action_type_lst:
                continue

            logging.info(data['name'][i])

            feature = data['feature'][i]
            self.location = data['location']
            if len(self.location) > 0:
                self.location = self.location[i]
            self.data = data['data'][i]
            self.label = data['label']
            weight = data['weight']
            out = data['out']
            cm = data['cm']
            try:
                self.cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            except Exception as e:
                self.cm = None

            dataset = self.args.dataset.split('-')[0]
            self.names = self.action_names[dataset]
            self.font_size = self.font_sizes[dataset]

            self.pred = np.argmax(out, 1)

            self.pred_class = self.pred[i] + 1
            self.actural_class = self.label[i] + 1

            self.probablity = out[i, self.label[i]]

            self.result = np.einsum('kc,ctvm->ktvm', weight, feature)  # CAM method
            self.result = self.result[self.label[i]]

            #     "driveract": {'class': 34, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
            if self.args.dataset == "3mdad_downsampled1":
                self.show_3MDAD1_skeleton(i, data['name'][i])
            elif self.args.dataset == "3mdad_downsampled2":
                self.show_3MDAD2_skeleton(i, data['name'][i])
            elif self.args.dataset == "ebdd":
                self.show_EBDD_skeleton(i, data['name'][i])
            elif self.args.dataset == "driveract":
                self.show_driveract_skeleton(i, data['name'][i], action_type)

    def show_3MDAD_skeleton3(self, i, sample_name):
        sample_name = str(sample_name).split(".")[0]
        base_img_path = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_imgs"
        import os
        imgs_folder_path = os.path.join(base_img_path, sample_name)
        if not os.path.exists(imgs_folder_path):
            sys.exit(-1)

        imgs_name_lst = os.listdir(imgs_folder_path)
        imgs_name_lst.sort(key=lambda x: int(x.split(".")[0][(str(x).index("F")) + 1:]))

        linewidth = 1.8
        jointwidth = 150
        SpecialJointWidth = 75

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 0, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        plt.figure(dpi=500)
        for t in list(range(T))[::3]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            plt.cla()
            W, H = 650, 480
            plt.xlim((0, W))
            plt.ylim((0, H))

            plt.axis('off')
            plt.title(
                'sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class[0-15]:{}, actural_class:{}'.format(
                    i, self.names[self.label[i]],
                    t, self.probablity * 100, self.pred[i], self.label[i]
                ))

            for m in range(M):
                x = self.location[0, t, :, m]
                y = H - self.location[1, t, :, m]
                c = []
                for v in range(V):
                    r = result[t, v, m]
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    plt.plot([x[v], x[k]], [y[v], y[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    plt.scatter(x[v], y[v], marker='o', color=[r, g, b], s=jointwidth * 0.2 * (1 + r * 15))

            import os
            out_images_folder_path = r"N:\3MDAD\Visual3MDAD\{}".format(sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)

    def show_3MDAD_skeleton2(self, i, sample_name):
        sample_name = str(sample_name).split(".")[0]
        # base_img_path = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_imgs"
        import os
        # imgs_folder_path = os.path.join(base_img_path, sample_name)
        # if not os.path.exists(imgs_folder_path):
        #     sys.exit(-1)

        # imgs_name_lst = os.listdir(imgs_folder_path)
        # imgs_name_lst.sort(key=lambda x: int(x.split(".")[0][(str(x).index("F")) + 1:]))

        linewidth = 1.8
        jointwidth = 150
        SpecialJointWidth = 75

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 0, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        plt.figure(dpi=500)
        for t in list(range(T))[::3]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            r_lst = {}
            plt.cla()
            W, H = 650, 480
            plt.xlim((0, W))
            plt.ylim((0, H))

            # img_path = os.path.join(imgs_folder_path, imgs_name_lst[t])
            from matplotlib.image import imread
            # img = imread(img_path)

            # from PIL import Image
            # img = Image.open(img_path)
            # img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # plt.imshow(img)

            # plt.axis('off')
            plt.title(
                'sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class[0-15]:{}, actural_class:{}'.format(
                    i, self.names[self.label[i]],
                    t, self.probablity * 100, self.pred[i], self.label[i]
                ))

            for m in range(M):
                x = self.location[0, t, :, m]
                y = H - self.location[1, t, :, m]
                c = []
                for v in range(V):
                    r = result[t, v, m]
                    r_lst[v] = r
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    plt.plot([x[v], x[k]], [y[v], y[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    # plt.scatter(x[v], y[v], marker='o', color=[r, g, b], s=jointwidth * 0.2 * (1 + r * 15),
                    #             edgecolors="red", linewidths=10)
                    if r >= 0.1:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="red", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
                    else:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="blue", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
            # r_lst
            print("{} ---- {}".format(t, r_lst))

            import os
            out_images_folder_path = r"N:\3MDAD\Visual3MDAD2\{}".format(sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)

    def show_3MDAD1_skeleton(self, i, sample_name):
        sample_name = str(sample_name).split(".")[0]
        base_img_path = r"N:\3MDAD\Day\RGB1_mix"
        import os
        imgs_folder_path = os.path.join(base_img_path, sample_name)
        if not os.path.exists(imgs_folder_path):
            sys.exit(-1)

        imgs_name_lst = os.listdir(imgs_folder_path)
        imgs_name_lst.sort(key=lambda x: int(x.split(".")[0][(str(x).index("F")) + 1:]))

        linewidth = 1.8

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        plt.figure(dpi=500)
        for t in list(range(T))[::3]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            r_lst = {}
            plt.cla()
            W, H = 640, 480
            plt.xlim((0, W))
            plt.ylim((0, H))
            try:
                img_path = os.path.join(imgs_folder_path, imgs_name_lst[t])
                from matplotlib.image import imread
            except Exception as e:
                continue

            from PIL import Image
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            plt.imshow(img)

            plt.axis('off')

            for m in range(M):
                x = self.location[0, t, :, m]
                y = H - self.location[1, t, :, m]
                c = []
                for v in range(V):
                    r = result[t, v, m]
                    r_lst[v] = r
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    plt.plot([x[v], x[k]], [y[v], y[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    if r >= 0.1:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="red", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
                    else:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="blue", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
            # r_lst
            print("{}---{} ---- {}".format(sample_name, t, r_lst))
            import os
            out_images_folder_path = os.path.join(self.args.out_images_folder_path, sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)

    def show_3MDAD2_skeleton(self, i, sample_name):
        sample_name = str(sample_name).split(".")[0]
        base_img_path = r"N:\3MDAD\Day\RGB2_pose_220104\softnode3_imgs"
        import os
        imgs_folder_path = os.path.join(base_img_path, sample_name)
        if not os.path.exists(imgs_folder_path):
            sys.exit(-1)

        imgs_name_lst = os.listdir(imgs_folder_path)
        imgs_name_lst.sort(key=lambda x: int(x.split(".")[0][(str(x).index("F")) + 1:]))

        linewidth = 1.8

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        plt.figure(dpi=500)
        for t in list(range(T))[::3]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            r_lst = {}
            plt.cla()
            W, H = 640, 480
            plt.xlim((0, W))
            plt.ylim((0, H))
            try:
                img_path = os.path.join(imgs_folder_path, imgs_name_lst[t])
                from matplotlib.image import imread
            except Exception as e:
                continue

            from PIL import Image
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            plt.imshow(img)

            plt.axis('off')

            for m in range(M):
                x = self.location[0, t, :, m]
                y = H - self.location[1, t, :, m]
                c = []
                for v in range(V):
                    r = result[t, v, m]
                    r_lst[v] = r
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    plt.plot([x[v], x[k]], [y[v], y[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    if r >= 0.1:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="red", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
                    else:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="blue", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
            # r_lst
            print("{}---{} ---- {}".format(sample_name, t, r_lst))
            import os
            out_images_folder_path = os.path.join(self.args.out_images_folder_path, sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)

    def show_EBDD_skeleton(self, i, sample_name):
        sample_name = str(sample_name).split(".")[0]
        base_img_path = r"N:\clip225\clip225"
        import os
        imgs_folder_path = os.path.join(base_img_path, sample_name)
        if not os.path.exists(imgs_folder_path):
            sys.exit(-1)

        imgs_name_lst = os.listdir(imgs_folder_path)
        imgs_name_lst.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        linewidth = 1.8

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 0, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        plt.figure(dpi=500)
        during = 10
        for t in list(range(T))[::during]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            r_lst = {}
            plt.cla()
            W, H = 453, 253
            plt.xlim((0, W))
            plt.ylim((0, H))
            try:
                img_path = os.path.join(imgs_folder_path, imgs_name_lst[t])
                from matplotlib.image import imread
            except Exception as e:
                continue

            from PIL import Image
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            plt.imshow(img)

            plt.axis('off')

            for m in range(M):
                x = self.location[0, t, :, m]
                y = H - self.location[1, t, :, m]
                c = []
                for v in range(V):
                    r = result[t, v, m]
                    r_lst[v] = r
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    plt.plot([x[v], x[k]], [y[v], y[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    # plt.scatter(x[v], y[v], marker='o', color=[r, g, b], s=jointwidth * 0.2 * (1 + r * 15),
                    #             edgecolors="red", linewidths=10)
                    if r >= 0.1:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="red", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
                    else:
                        plt.scatter(x[v], y[v], marker='o', edgecolors="blue", linewidths=12, color=[0.1, 0.1, 0.1],
                                    s=80)
            # r_lst
            print("{}---{} ---- {}".format(sample_name, t, r_lst))

            import os
            out_images_folder_path = os.path.join(self.args.out_images_folder_path, sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)

    def show_driveract_skeleton(self, i, sample_name, action_type):
        sample_name = str(sample_name).split(".")[0]

        linewidth = 1.8

        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 0, 0, 8, 9])

        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        fig = plt.figure(dpi=500)
        ax3d = fig.add_subplot(projection='3d')

        during = 5
        for t in list(range(T))[::during]:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            r_lst = {}
            plt.cla()
            W, H = 1, 2
            # plt.xlim((-W, W))
            # plt.ylim((0, 2 * H))

            # plt.axis('off')

            for m in range(M):
                x = self.location[0, t, :, m]
                y = - self.location[1, t, :, m]
                z = 1 - self.location[2, t, :, m]

                c = []
                for v in range(V):
                    r = result[t, v, m]
                    r_lst[v] = r
                    g = 0
                    b = 1 - r
                    c.append([r, g, b])
                    k = connecting_joint[v]
                    if x[v] == 0.0 or x[k] == 0.0:
                        continue

                    ax3d.plot([x[v], x[k]], [y[v], y[k]], [z[v], z[k]], '-', color=[0.1, 0.1, 0.1], linewidth=linewidth)
                    if r >= 0.1:
                        ax3d.scatter(x[v], y[v], z[v], marker='o', edgecolors="red", linewidths=12,
                                     color=[0.1, 0.1, 0.1],
                                     s=80)
                    else:
                        ax3d.scatter(x[v], y[v], z[v], marker='o', edgecolors="blue", linewidths=12,
                                     color=[0.1, 0.1, 0.1],
                                     s=80)
            # r_lst
            print("{}---{} ---- {}".format(sample_name, t, r_lst))

            import os
            out_images_folder_path = os.path.join(self.args.out_images_folder_path, str(action_type))
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)
            out_images_folder_path = os.path.join(out_images_folder_path, sample_name)
            if not os.path.exists(out_images_folder_path):
                os.mkdir(out_images_folder_path)

            plt.savefig(os.path.join(out_images_folder_path, "i{}.png".format(t)), format='png', bbox_inches='tight',
                        pad_inches=0)
