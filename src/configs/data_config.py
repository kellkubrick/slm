
from easydict import EasyDict


data_cfg = EasyDict()

data_cfg.tiny_stories_dataset = EasyDict()
data_cfg.tiny_stories_dataset.name = 'TinyStoriesDataset'
data_cfg.tiny_stories_dataset.path_to_data = 'src/data/dataset'
data_cfg.tiny_stories_dataset.vocabulary_size = 30000
data_cfg.tiny_stories_dataset.sort = True

data_cfg.tiny_stories_dataset.preprocessing = EasyDict()
data_cfg.tiny_stories_dataset.preprocessing.raw_data_path_template = 'raw_data_%s_%s.txt'  # set type, lang name
data_cfg.tiny_stories_dataset.preprocessing.tokenizer_path = 'tokenization_%s.pickle'  # lang name
data_cfg.tiny_stories_dataset.preprocessing.preprocessed_data_path_template = 'tokenized_data_%s_%s.pickle'  # set type, lang name
data_cfg.tiny_stories_dataset.preprocessing.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
data_cfg.tiny_stories_dataset.preprocessing.end_of_word = '</w>'
data_cfg.tiny_stories_dataset.preprocessing.min_frequency = 2

print(data_cfg)