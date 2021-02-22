import os
#os.environ['CUDA_VISIBLE_DEVICE'] = '1'

import argparse

from torch.utils.data import DataLoader

from model import Document_Transformer, Researcher_Transformer
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab

import pickle
import torch

def dump_data(obj, wfpath):
    with open(rfpath, 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath):
    with open(rfpath, 'rb') as rf:
        return pickle.load(rf)

def train(args):

    with open(args.train_dataset, "r", encoding='utf-8') as f:
        vocab = WordVocab(f, args.wordemb)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab('output/output.vocab')

    #print("Loading Vocab", args.vocab_path)
    #vocab = WordVocab.load_vocab(args.vocab_path)
    #print("Vocab Size: ", len(vocab))

    author_community = load_data(args.author_community)
    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, author_community, vocab, seq_len=args.seq_len, doc_len = args.doc_len, hops = args.hops,
                                corpus_lines=args.corpus_lines)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Building RPT model")
    document_transformer = Document_Transformer(vocab.vectors, len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    researcher_transformer = Researcher_Transformer(hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating RPT Trainer")
    trainer = BERTTrainer(document_transformer, researcher_transformer, len(vocab), train_dataloader=train_data_loader, 
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)
    print("Training Start")

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="author publication for train bert")
    parser.add_argument("-ac", "--author_community", required=True, type=str, help="author community for train bert")
    parser.add_argument("-we", "--wordemb", type=str, default=None, help="pre-trained word embeddings")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("-d", "--doc_len", type=int, default=10, help="maximum document number")
    parser.add_argument("-hp", "--hops", type=int, default=2, help="community hops")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=10, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=50, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-7, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    train(args)
