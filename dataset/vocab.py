import abc
import pickle
from collections import Counter


class TorchVocab(object):
    """
    Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens in the data used to build the Vocab.
        stoi: A collections.default_dict instance mapping token strings to numerical identifiers.
            Key is the token string.
            Value is the index of the token string.
        itos: A list of token strings indexed by their numerical identifiers.
            Value is the token string.
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=None, vectors=None, unk_init=None,
                 vectors_cache=None):
        """
        Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no maximum.
                Default: None.
            min_freq: The minimum frequency needed to include a token in the vocabulary.
                Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that will be prepended to the vocabulary
                in addition to an <unk> token. Default: ['<pad>', '<unk>'].
            vectors: One of either the available pretrained vectors or custom pretrained vectors
                (see Vocab.load_vectors).
            unk_init (callback): by default, initialize out-of-vocabulary word vectors to zero vectors;
                can be any function that takes in a Tensor and returns a Tensor of the same size. Default: None.
            vectors_cache: directory for cached vectors. Default: '.vector_cache'.
        """
        if specials is None:
            specials = ['<pad>', '<unk>']   # <pad> for padding, <unk> for unknown words

        self.freqs = counter
        counter = counter.copy()    # copy counter to avoid changing it
        min_freq = max(min_freq, 1)

        self.itos = list(specials)  # itos is a list that holds the string of each token.

        # frequencies of special tokens are not counted when building vocabulary in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        word_and_freq = sorted(counter.items(), key=lambda tup: tup[0])  # sort by alphabetically, key is token string
        word_and_freq.sort(key=lambda tup: tup[1], reverse=True)     # sort by frequency, descending order

        for tok, freq in word_and_freq:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(tok)

        # stoi is a dictionary that holds the index of each token in the vocabulary
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}     # key is token string, value is index

        self.vectors = None
        if vectors is not None:
            assert False, "Need to implement load_vectors() function."
            # self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        """
        Compare two TorchVocab objects with each other using "=="

        Arguments:
            other: object to compare
        Return:
            boolean value
        """
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False

    def __len__(self):
        """
        Return the length of the vocabulary.

        Return:
            length of the vocabulary
        """
        return len(self.itos)

    def vocab_rerank(self):
        """ Re-rank the vocabulary. """
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def extend(self, v, sort=False):
        assert isinstance(v, TorchVocab)
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    """
    Subclass of TorchVocab.

    Attributes:
        stoi: A collections.default_dict instance mapping token strings to numerical identifiers.
            Key is the token string.
            Value is the index of the token string.
        itos: A list of token strings indexed by their numerical identifiers.
            Value is the token string.
    """
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0   # pad_index is the index of <pad> token, meaning the index of padding token is 0
        self.unk_index = 1   # unk_index is the index of <unk> token, meaning the index of unknown word token is 1
        self.eos_index = 2   # eos_index is the index of <eos> token, meaning the index of end of sentence token is 2
        self.sos_index = 3   # sos_index is the index of <sos> token, meaning the index of start of sentence token is 3
        self.mask_index = 4  # mask_index is the index of <mask> token, meaning the index of mask token is 4
        specials = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        super().__init__(counter, specials=specials, max_size=max_size, min_freq=min_freq)

    @abc.abstractmethod
    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    @abc.abstractmethod
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)


