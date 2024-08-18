from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
WeightsInitType = IntEnum(
    'WeightsInitType', ('normal', 'uniform', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
)
InferenceType = IntEnum('InferenceType', ('greedy', 'temperature'))
ImageAugmentationType = IntEnum('ImageAugmentationType', ('base', 'deit_3'))