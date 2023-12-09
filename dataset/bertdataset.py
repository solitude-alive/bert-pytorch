from torch.utils.data import Dataset
import tqdm
import random
import torch


class BERTDataset(Dataset):
    """
    BERT Dataset

    """
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.corpus_lines = corpus_lines
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1
            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            min_lines = 1000
            for _ in range(random.randint(min(self.corpus_lines, min_lines), self.corpus_lines)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        # add pad token, pad_index is 0, means not to predict
        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]  # 1 for t1, 2 for t2

        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]  # pad to seq_len

        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label (isNotNext: 0, isNext: 1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def random_word(self, sentence):
        """
        Randomly mask words in sentence

        param:
            sentence: list of int, tokenized sentence

        return:
            tokens: list of int, tokenized sentence with random masked words
            output_label: list of int, 1 means to predict, 0 means not to predict
        """
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()  # sample in uniform [0, 1)
            if prob < 0.15:
                prob /= 0.15    # use sub prob to decide which action to take

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token, prob between 0.8 and 0.9
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly not change token, but need to be predicted
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab[token])  # 1 means to predict
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)  # 0 means not to predict

        return tokens, output_label

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
            t1, t2 = line[:-1].split("\t")

            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            min_lines = 1000
            for _ in range(random.randint(min(self.corpus_lines, min_lines), self.corpus_lines)):
                self.random_file.__next__()
            line = self.random_file.__next__()

        return line[:-1].split("\t")[1]
