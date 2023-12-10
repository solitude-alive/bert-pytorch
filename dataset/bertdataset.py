from torch.utils.data import Dataset
import torch
import json


class BERTDataset(Dataset):
    """
        BERT Dataset

        """
    def __init__(self, corpus_path):
        self.corpus_lines = None
        with open(corpus_path) as f:
            self.corpus_lines = f.readlines()

    def __len__(self):
        return len(self.corpus_lines)

    def __getitem__(self, item):
        line = json.loads(self.corpus_lines[item])

        bert_input = line["bert_input"]
        bert_label = line["masked_labels"]
        segment_label = line["segment_ids"]
        is_next_label = line["next_sentence_labels"]
        mask_weights = line["masked_lm_weights"]

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "mask_weights": mask_weights
                  }

        return {key: torch.tensor(value) for key, value in output.items()}
