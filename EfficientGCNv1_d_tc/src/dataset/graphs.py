import logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph():
    def __init__(self, dataset, Ad=False, withself=False, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency(A_d=Ad, with_self=withself)
        print("===>A:", np.sum(self.A))

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
        elif self.dataset == '3mdad' or self.dataset == "3mdad_downsampled" or self.dataset == "dad" or str(
                self.dataset).lower() == "ebdd" or self.dataset == "driveract":
            num_node = 12
            neighbor_link = [(0, 1), (2, 1), (3, 2),
                             (4, 3), (5, 1), (6, 5), (7, 6),
                             (8, 0), (9, 0), (10, 8), (11, 9)]
            connect_joint = np.array([1, 1, 1, 2, 3,
                                      1, 5, 6,
                                      0, 0, 8, 9])  # 12
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([0, 1])  # torso
            ]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self, A_d=False, with_self=False):
        hop_dis = self._get_hop_distance()
        # valid_hop = range(0, self.max_hop + 1, self.dilation)
        # adjacency = np.zeros((self.num_node, self.num_node))
        # for hop in valid_hop:
        #     adjacency[hop_dis == hop] = 1
        # normalize_adjacency = self._normalize_digraph(adjacency)
        # A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        # for i, hop in enumerate(valid_hop):
        #     A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        # return A

        self.hop_dis = hop_dis
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))

        for i, hop in enumerate(valid_hop):
            if A_d == False:
                adjacency = np.zeros((self.num_node, self.num_node))
                for hop_ in valid_hop:
                    adjacency[self.hop_dis == hop_] = 1
                self.adjacency = adjacency
                self.normalize_adjacency = self._normalize_digraph(adjacency)
                A[i][self.hop_dis == hop] = self.normalize_adjacency[self.hop_dis == hop]
                if with_self == True:
                    if i != 0:
                        A[i] += A[0]
            else:
                adjacency = np.zeros((self.num_node, self.num_node))
                if with_self == True:
                    adjacency[self.hop_dis == 0] = 1
                adjacency[self.hop_dis == hop] = 1
                normalize_adjacency = self._normalize_digraph(adjacency)
                A[i] = normalize_adjacency

        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
