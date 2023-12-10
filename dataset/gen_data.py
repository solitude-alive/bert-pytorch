"""
This script is used to generate the data for the project.
"""

import json
import os
import tqdm
from dataset.tokenization import FullTokenizer


def load_text(dic):
    """
    Load the txt from the dictionary.
    """
    return dic["text"]


def gen_gpt():
    folder = "../data/json"
    folder_out = "../data/txt"

    for file in os.listdir(folder):
        if file.endswith(".jsonl"):
            print(file)
            # file = "small-117M.test.jsonl"
            # replace the extension with .txt
            txt_file = file.replace(".jsonl", ".txt")

            # open the file and write the text only
            with open(os.path.join(folder_out, txt_file), "w") as ft:
                # open the jsonl file
                with open(os.path.join(folder, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_dict = json.loads(line)  # convert to dict
                        text = load_text(line_dict)  # get the text only
                        text = text.replace("\n", " ")  # remove new line,
                        # insert the \t in the middle of the text
                        text = text[: len(text) // 2] + "\t" + text[len(text) // 2:]
                        ft.write(text + "\n")


def gen_wiki():
    folder = "../data/wikitext-103-raw"
    folder_out = "../data/txt"

    tk = FullTokenizer("vocab.txt")

    for file in os.listdir(folder):
        if file.endswith(".raw"):
            print(file)
            txt_file = file.replace(".raw", ".txt")

            # open the file and write the text only
            with open(os.path.join(folder_out, txt_file), "w") as ft:
                # open the jsonl file
                with open(os.path.join(folder, file), "r") as f:
                    lines = f.readlines()
                    for line in tqdm.tqdm(lines,
                                          total=len(lines),
                                          bar_format="{l_bar}{r_bar}"
                                          ):
                        if line.strip() == '' or line.strip()[0] == '=':
                            continue
                        text = line
                        text = " ".join(tk.tokenize(text))    # tokenize the text
                        # insert the \t in the middle of the text
                        text = text[: len(text) // 2] + "\t" + text[len(text) // 2:]
                        ft.write(text + "\n")


if __name__ == "__main__":
    gen_wiki()

