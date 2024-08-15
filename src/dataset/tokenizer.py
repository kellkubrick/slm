import os
import pickle
import re
from collections import defaultdict
from typing import Union


class BPETokenizer:
    """A class for Byte-Pair Encoding (BPE) tokenizer implementation.

    This tokenization algorithm originally proposed in
        'Neural Machine Translation of Rare Words with Subword Units' (https://aclanthology.org/P16-1162.pdf)
    """

    def __init__(self, special_tokens: list[str], end_of_word: str, vocabulary_size: int, lowercase: bool = False):
        """Tokenizer initialization.

        Args:
            special_tokens: special tokens list ([SOS], [EOS], [PAD] etc)
            end_of_word: a special character sequence representing the end of a word
            vocabulary_size: desired vocabulary size
            lowercase: if to lowercase training texts
        """
        self.special_tokens = special_tokens
        self.end_of_word = end_of_word
        self.use_unk_token = '[UNK]' in special_tokens
        self.use_bounds_tokens = '[SOS]' in special_tokens and '[EOS]' in special_tokens
        self.vocabulary_size = vocabulary_size
        self.lowercase = lowercase

        self._init_vocabulary()

    def _init_vocabulary(self, id2token=None, token2id=None):
        """Initializes vocabulary."""
        if all((id2token, token2id)):
            self.id2token, self.token2id = id2token, token2id
        else:
            tokens_to_add = self.special_tokens + list(map(str, range(10)))

            self.id2token = {i: token for i, token in enumerate(tokens_to_add)}
            self.token2id = {token: i for i, token in enumerate(tokens_to_add)}

    def get_words(self, file_path: str) -> dict:
        """Gets a word counter based on the training text corpus.

        Args:
            file_path: a path to the text file where each raw represents one training sequence (sentence)

        Returns:
            A dictionary with keys represented as words (as tuples of characters) and values as word frequencies
        """
        word_counts = defaultdict(int)
        with open(file_path, 'r', encoding='utf-8') as f:
            for seq in f:
                if self.lowercase:
                    seq = seq.lower().strip()
                else:
                    seq = seq.strip()
                seq = re.sub(r'\d+', '', seq)
                for word in seq.split():
                    word_counts[word] += 1
            word_counts = dict([(tuple(x[:-1]) + (x[-1] + self.end_of_word,), y) for (x, y) in word_counts.items()])
        return word_counts

    @staticmethod
    def prepare_tokens(word_counts: dict) -> list:
        """Gets tokens from the words counter.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies

        Returns:
            A sequence with tokens sorted in reverse order by length
        """
        # TODO: Gather all unique tokens from word_counts in one set and return it. This set will then be used
        #           to construct the vocabulary mappings
        tokens = set()
        for word in word_counts.keys():
            tokens.update(word)
        return sorted(tokens, key=len, reverse=True)

    @staticmethod
    def get_pair_statistics(word_counts: dict) -> dict:
        """Gets pairs frequency statistics.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies

        Returns:
            A dictionary with keys as a tuples of token pairs and values as these pairs frequency
        """

        #
        pairs = defaultdict(int)
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def merge(word_counts: dict, most_frequent_pair: tuple) -> dict:
        """Makes tokens merging step.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies
            most_frequent_pair: a pair of the most frequently occurring tokens

        Returns:
            A dictionary with keys represented as words (as merged tokens) and values as word frequencies
        """
        # TODO: To implement merging step:
        #       For each word:
        #           1. Replace a pair of tokens from most_frequent_pair with one new merged token (if this pair is presented in a word)
        #           2. Get new tuple of tokens representing the word and update new_vocab using this tuple as a key
        #                   and word frequency from word_counts as a value
        #       Return filled new_vocab
        new_vocab = {}
        start_char, stop_char = most_frequent_pair
        str_pair = "".join(most_frequent_pair)
        pattern = re.compile(r'(?<!\S)' + re.escape(start_char + ' ' + stop_char) + r'(?!\S)')

        for word, freq in word_counts.items():
            new_word = ' '.join(word)
            new_word = pattern.sub(str_pair, new_word)
            new_word = tuple(new_word.split(' '))
            new_vocab[new_word] = freq
        return new_vocab

    def _update_vocabulary(self, word_counts: dict):
        """Updates vocabulary with new tokens.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies
        """
        # TODO: Updated vocabulary:
        #        1) gather tokens with self.prepare_tokens() and add all new tokens to self.id2token
        #           and self.token2id mappings. You can use iterating index as "id". Don't forget that vocabulary
        #           mappings have been already initialized with some tokens, so new token ids must be shifted!
        #        2) add all unique symbols (including end of word) that are present in the training set
        tokens = BPETokenizer.prepare_tokens(word_counts)
        shift = len(self.token2id)
        for i, token in enumerate(tokens):
            if token not in self.token2id:
                self.token2id[token] = shift + i
                self.id2token[shift + i] = token

        uniq_symbols = set(''.join(tokens))
        shift, added = len(self.id2token), 0
        for uniq_symbol in uniq_symbols:
            if uniq_symbol not in self.token2id:
                self.id2token[added + shift] = uniq_symbol
                self.token2id[uniq_symbol] = added + shift
                added += 1

        self.id2token[added + shift] = self.end_of_word
        self.token2id[self.end_of_word] = added + shift

    def train(self, file_path: str):
        """Makes BPE training.

        Args:
            file_path: a path to the text file where each raw represents one training sequence (sentence)
        """
        # TODO: Implement BPE training process:
        #       1. Get all training corpus words using self.get_words() method
        #       2. Make loop for a number of iterations equal to self.vocabulary_size and at each step:
        #               a) get token pairs statistics with self.get_pair_statistics()
        #               b) find the most frequently occurring token pair
        #               c) make vocabulary merging step with self.merge() and found most frequent token pair
        #       3. Update vocabulary with self._update_vocabulary and merged words counter
        words = self.get_words(file_path)

        for i in range(self.vocabulary_size):
            pair_stat = self.get_pair_statistics(words)
            most_freq = max(pair_stat, key=pair_stat.get)
            words = self.merge(words, most_freq)

        self._update_vocabulary(words)

    def encode(self, text: str, tokenize=False) -> list[int]:
        """Encodes text into token ids.

        Args:
            text: a text sequence to encode

        Returns:
            A list of tokens for input text
        """
        # TODO: Encode text by replacing vocabulary tokens with ids using self.token2id mapping
        #   1. Preprocess the text:
        #       - Convert text to lowercase if self.lowercase is True.
        #       - Split the text into words and add end_of_word token to each word.
        #   2. Add special tokens:
        #       - If self.use_bounds_tokens is True, add [SOS] at the beginning and [EOS] at the end of the word list.
        #   3. Encode the text:
        #       - Iterate through each word and its subwords.
        #       - Replace vocabulary tokens with ids using self.token2id mapping.
        #       - If the length of a subword is 1 and it is not found in self.token2id and self.use_unk_token is True, use the [UNK] token.
        #   4. Return the list of encoded token ids.
        tokens = []
        encoded = []
        if self.lowercase:
            text = text.lower()
        words = [word+self.end_of_word for word in text.strip().split()]

        if self.use_bounds_tokens:
            words = ['[SOS]'] + words + ['[EOS]']

        for word in words:
            i = 0
            while i < len(word):
                unknown = True
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.token2id:
                        tokens.append(subword)
                        encoded.append(self.token2id[subword])
                        i = j - 1
                        unknown = False
                        break
                i += 1
                if unknown and self.use_unk_token:
                    encoded.append(self.token2id['[UNK]'])
                    tokens.append('[UNK]')
        print(tokens)
        if tokenize:
            print(encoded)
            return encoded

        return tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes token ids into text.

        Args:
            tokens: a list of token ids
            skip_special_tokens: a boolean indicating whether to drop special tokens from decoded sequence

        Returns:
            Decoded text
        """
        # TODO: Decode token ids by replacing them with text using self.id2token
        #       Replace special tokens (including end_of_word) with a space and strip() the result
        #               if skip_special_tokens is True
        decoded = ''.join([self.id2token[t] for t in tokens])
        print(decoded)
        if self.use_bounds_tokens and not skip_special_tokens:
            decoded = decoded.replace('[SOS]', f'[SOS]{self.end_of_word}').replace('[EOS]', f'[EOS]{self.end_of_word}')

        if skip_special_tokens:
            decoded = re.sub('|'.join(map(re.escape, self.special_tokens)), ' ', decoded).strip()
            print(decoded)

        return decoded.replace(self.end_of_word, ' ').strip()

    def save(self, path: str):
        """Saves vocabulary state."""
        with open(path, 'wb') as f:
            pickle.dump({'id2token': self.id2token, 'token2id': self.token2id}, f)

    def load(self, path):
        """Loads vocabulary state."""
        with open(path, 'rb') as f:
            vocab_mappings = pickle.load(f)

        self._init_vocabulary(vocab_mappings['id2token'], vocab_mappings['token2id'])

    def get_vocab_size(self) -> int:
        """Gets vocabulary size."""
        return len(self.id2token)