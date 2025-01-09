import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder
from .dad import DAD_Feeder, M3DADRGB2_Location_Feeder, EBDDRGB2_Location_Feeder, M3DADRGB1_Location_Feeder, \
    DriveAct_Location_Feeder

__data_args = {
    'ntu-xsub': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xview': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xsub120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xset120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},

    '3mdad': {'class': 16, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
    'dad': {'class': 9, 'shape': [3, 6, 225, 12, 1], 'feeder': DAD_Feeder},
    'ebdd': {'class': 5, 'shape': [3, 6, 225, 12, 1], 'feeder': DAD_Feeder},
    '3mdad_downsampled1': {'class': 16, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
    '3mdad_downsampled2': {'class': 16, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
    "driveract": {'class': 32, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
}


def create(dataset, root_folder, transform, num_frame, inputs, **kwargs):
    graph = Graph(dataset, Ad=kwargs["Ad"], withself=kwargs["withself"])
    try:
        data_args = __data_args[dataset]
        data_args['shape'][0] = len(inputs)
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    if transform:
        dataset_path = '{}/transformed/{}'.format(root_folder, dataset)
    else:
        dataset_path = root_folder
    kwargs.update({
        'dataset_path': dataset_path,
        'inputs': inputs,
        'num_frame': num_frame,
        'connect_joint': graph.connect_joint,
        "suffix": kwargs["suffix"],
    })
    feeders = {
        'train': data_args['feeder']('train', **kwargs),
        'eval': data_args['feeder']('eval', **kwargs),
    }
    if 'ntu' in dataset:
        print("Location: {}".format("NTU_Location_Feeder"))
        feeders.update({'location': NTU_Location_Feeder(data_args['shape'])})
    elif "3mdad_downsampled2" == dataset:
        print("Location: {}".format("M3DADRGB2_Location_Feeder"))
        feeders.update({'location': M3DADRGB2_Location_Feeder(data_args['shape'])})
    elif "3mdad_downsampled1" == dataset:
        print("Location: {}".format("M3DADRGB1_Location_Feeder"))
        feeders.update({'location': M3DADRGB1_Location_Feeder(data_args['shape'])})
    elif "ebdd" == dataset:
        print("Location: {}".format("EBDDRGB2_Location_Feeder"))
        feeders.update({'location': EBDDRGB2_Location_Feeder(data_args['shape'])})
    elif "driveract" == dataset:
        print("Location: {}".format("DriveAct_Location_Feeder"))
        feeders.update({'location': DriveAct_Location_Feeder(data_args['shape'])})

    return feeders, data_args['shape'], data_args['class'], graph.A, graph.parts
