from tokenizer import BPETokenizer
from src.models.embeddings import Embedding, PositionalEncoding


class Preprocessor:
    def __init__(self, config, data_path):
        self.config = config
        self._init_tokenizer()
        self._init_embedder()
        self._init_pe()

    def _init_tokenizer(self):
        self.tokenizer = BPETokenizer(self.config.special_tokens, self.config.end_of_word, self.config.vocab_size)

    def _init_embedder(self):
        self.embedder = Embedding(self.config.vocab_size, self.config.d_model)

    def _init_pe(self):
        self.pe = PositionalEncoding()

    def train_tokenizer(self):
        pass

    def preprocess(self, text):
        pass