import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder
from .dad import DAD_Feeder

__data_args = {
    '3mdad': {'class': 16, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},
    'dad': {'class': 9, 'shape': [3, 6, 225, 12, 1], 'feeder': DAD_Feeder},
    'ebdd': {'class': 5, 'shape': [3, 6, 225, 12, 1], 'feeder': DAD_Feeder},
    '3mdad_downsampled': {'class': 16, 'shape': [3, 6, 50, 12, 1], 'feeder': DAD_Feeder},

    "driveract": {'class': 32, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
    "driveract_task": {'class': 12, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
    "driveract_all": {'class': 372, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
    "driveract_action": {'class': 6, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
    "driveract_object": {'class': 17, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
    "driveract_location": {'class': 14, 'shape': [3, 6, 90, 12, 1], 'feeder': DAD_Feeder},
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
        'test': data_args['feeder']('test', **kwargs),
    }
    samples_weight = feeders["train"].get_samples_weight()

    return feeders, data_args['shape'], data_args['class'], graph.A, graph.parts, samples_weight
