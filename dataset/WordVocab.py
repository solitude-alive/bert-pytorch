import pickle
from collections import Counter
import tqdm
from dataset.vocab import Vocab
from utils import logger


# Build Vocab with text file
class WordVocab(Vocab):
    """
    Attributes:
        stoi: A collections.default_dict instance mapping token strings to numerical identifiers.
            Key is the token string.
            Value is the index of the token string.
        itos: A list of token strings indexed by their numerical identifiers.
            Value is the token string.
    """

    def __init__(self, texts, max_size=None, min_freq=1):
        logger.info("Start building vocab...")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()
            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False, with_len=False) -> list:
        """ convert word sequence to index sequence """
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq  # add <sos> at the beginning
        if with_eos:
            seq = seq + [self.eos_index]  # add <eos> at the end

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq = seq + [self.pad_index] * (seq_len - len(seq))
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        """ Convert index sequence to word sequence """
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path) -> "WordVocab":
        """ load vocab from file """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab


def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", type=str, default="sample_txt.txt")
    parser.add_argument("-o", "--output_path", type=str, default="vocab.pkl")
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    logger.info("VOCAB SIZE:", len(vocab))

    vocab.save_vocab(args.output_path)
    logger.info("Saved vocab file to '%s'" % args.output_path)

    vocab_load = vocab.load_vocab(args.output_path)
    logger.info("Loaded vocab file from '%s'" % args.output_path)


if __name__ == "__main__":
    logger.configure(dir_log="../log/bert_pre", format_strs=["stdout", "log"])
    logger.log("test")
    build()
