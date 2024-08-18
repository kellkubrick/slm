from tokenizer import BPETokenizer
from src.models.embeddings import Embedding, PositionalEncoding


class Preprocessor:
    def __init__(self, config, data_path):
        self.config = config

        self.special_tokens = self.config.data_config.tiny_stories_dataset.preprocessing.special_tokens
        self.end_of_word = self.config.data_config.tiny_stories_dataset.preprocessing.end_of_word
        self.vocab_size = self.config.data_config.tiny_stories_dataset.vocabulary_size
        self.d_model = self.config.model_config.d_model
        self.max_seq_length = self.config.model_config.max_sequence_length
        self.dropout_rate = self.config.model_config.dropout_rate

        self._init_tokenizer()
        self._init_embedder()
        self._init_pe()

    def _init_tokenizer(self):
        self.tokenizer = BPETokenizer(self.special_tokens, self.end_of_word, self.vocab_size)

    def _init_embedder(self):
        self.embedder = Embedding(self.vocab_size, self.d_model)

    def _init_pe(self):
        self.pe = PositionalEncoding(self.max_seq_length, self.d_model, self.dropout_rate)

    def train_tokenizer(self):
        pass

    def preprocess(self, text):
        tokens = self.tokenizer.encode(text)
        embeddings = self.pe(self.embedder(tokens))
        return embeddings


if __name__ == "__main__":
    pass
