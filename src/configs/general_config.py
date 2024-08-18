from easydict import EasyDict
from data_config import data_cfg
from model_config import model_cfg
from train_config import experiment_cfg

general_config = EasyDict()

general_config.data_config = data_cfg
general_config.model_config = model_cfg
general_config.train_config = experiment_cfg

print(general_config.data_config.tiny_stories_dataset.preprocessing.special_tokens)

