from utils import logger
import argparse

from torch.utils.data import DataLoader

from model.bert import BERT
from trainer.pretrain import BERTTrainer
from dataset.bertdataset import BERTDataset
from dataset import tokenization


def train():
    logger.configure("bert", format_strs=["stdout", "log"], file_suffix=False)

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", type=str, help="output bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=1000, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    args.train_dataset = "data/data/wiki.train.raw"
    args.test_dataset = "data/data/wiki.test.raw"
    args.vocab_path = "dataset/vocab.txt"
    args.output_path = logger.get_dir()

    logger.log("Loading Vocab", args.vocab_path)
    vocab = tokenization.load_vocab(args.vocab_path)
    logger.log("Vocab Size: ", len(vocab))

    logger.log("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset)

    logger.log("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset) if args.test_dataset is not None else None

    logger.log("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    logger.log("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    logger.log("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    logger.log("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
