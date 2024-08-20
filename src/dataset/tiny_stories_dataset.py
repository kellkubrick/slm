import os

from torch.utils.data import Dataset

from dataset.preprocessing import Preprocessing
from utils.common_functions import write_file, read_file
from utils.enums import SetType


class TinyStoriesDataset(Dataset):
    """A class for Translation Dataset."""

    def __init__(self, config, set_type: SetType):
        self.config = config
        self.set_type = set_type

        self._init_preprocessors()
        self._get_data()
        self._sort_data()

    def encode_data(self, lang: str, tokenizer_path_to_load: str, preprocessed_data_path: str):
        """Encodes raw data to get numerical representation of text inputs.

        Args:
            lang: text language
            tokenizer_path_to_load: path to trained tokenizer state
            preprocessed_data_path: path to preprocessed data (final numerical representation)
        """
        # Train tokenizer
        self.preprocessors[lang].train(tokenizer_path_to_load)

        # Read raw data and encode it with trained tokenizer
        raw_data_path = os.path.join(
            self.config.path_to_data, self.config.preprocessing.raw_data_path_template % (self.set_type.name, lang)
        )
        raw_data = open(raw_data_path, encoding='utf-8').readlines()
        self.dataset[lang] = []

        for idx, text in enumerate(raw_data):
            tokens = self.preprocessors[lang].encode(text)
            self.dataset[lang].append({'id': idx, 'tokens': tokens})

        # Write encoded data
        write_file(self.dataset[lang], preprocessed_data_path)

    def _sort_data(self):
        """Sorts data by the number of tokens in samples if needed."""
        if not self.config.sort:
            return
        both_lang_samples = list(zip(self.dataset['en'], self.dataset['ru']))
        sorted_dataset = sorted(both_lang_samples, key=lambda pair: max(len(pair[0]['tokens']), len(pair[1]['tokens'])))
        self.dataset[self.config.source_lang], self.dataset[self.config.target_lang] = zip(*sorted_dataset)

    def _get_data(self):
        """Gets data to pass to Transformer model."""
        self.dataset = {}

        for lang in (self.config.source_lang, self.config.target_lang):
            if self.set_type == SetType.test and lang == self.config.target_lang:
                source_len = len(self.dataset[self.config.source_lang])
                self.dataset[lang] = [{'id': idx, 'tokens': []} for idx in range(source_len)]
                break

            preprocessed_data_path = os.path.join(
                self.config.path_to_data,
                self.config.preprocessing.preprocessed_data_path_template % (self.set_type.name, lang)
            )
            tokenizer_path_to_load = os.path.join(
                self.config.path_to_data, self.config.preprocessing.tokenizer_path % lang
            )

            if not os.path.exists(preprocessed_data_path):
                self.encode_data(lang, tokenizer_path_to_load, preprocessed_data_path)
            else:
                self.dataset[lang] = read_file(preprocessed_data_path)
                self.preprocessors[lang].load_tokenizer_state(tokenizer_path_to_load)

    def _init_preprocessors(self):
        """Initializes preprocessors for source and target languages."""
        self.preprocessors = {}

        for lang in (self.config.source_lang, self.config.target_lang):
            raw_data_path = os.path.join(
                self.config.path_to_data, self.config.preprocessing.raw_data_path_template % (SetType.train.name, lang)
            )
            self.preprocessors[lang] = Preprocessing(
                self.config.preprocessing, raw_data_path, self.config.vocabulary_size[lang]
            )

    def get_vocabulary_size(self):
        """Get data vocabulary size."""
        vocabulary_sizes = {}

        for lang in (self.config.source_lang, self.config.target_lang):
            vocabulary_sizes[lang] = self.preprocessors[lang].tokenizer.get_vocab_size()

        return vocabulary_sizes

    def __len__(self):
        return len(self.dataset[self.config.source_lang])

    def __getitem__(self, idx: int):
        sample_data = {
            'sample_pair_id': self.dataset[self.config.source_lang][idx]['id'],
            'source_lang_tokens': self.dataset[self.config.source_lang][idx]['tokens'],
            'target_lang_tokens': self.dataset[self.config.target_lang][idx]['tokens'],
        }
        return sample_data