import os
import argparse
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import time
from model import CodeModel
from utils import set_global_seeds

def parse_args():
    parser = argparse.ArgumentParser('CDC')

    # train/valid/test file
    parser.add_argument('--train_file', default='./data/embedding/train.pkl')
    parser.add_argument('--valid_file', default='./data/embedding/valid.pkl')
    parser.add_argument('--test_file', default='./data/embedding/test.pkl')
    
    # pretrined embedding
    parser.add_argument('--emb_file', default='./data/embedding/node_w2v_128')
    parser.add_argument('--no_pretrain', action='store_true',
                        help='whether use pretrained embs, Default is to use')
    parser.add_argument('--fix_emb', action='store_true',
                        help='whether fintune pretrained node embs, Default is to finetune')
    
    # train settings
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.,
                        help='dropout rate')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='size of the embeddings')
    parser.add_argument('--rnn_state_dim', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device, -1 means using cpu')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='the frequency of evaluating model on valid set') 
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_dir', default='./models')
    
    return parser.parse_args()


def main():
    set_global_seeds(2021)
    args = parse_args()
    
    train_data = pd.read_pickle(args.train_file)
    valid_data = pd.read_pickle(args.valid_file)
    test_data  = pd.read_pickle(args.test_file)
    
    word2vec = Word2Vec.load(args.emb_file).wv
    vocab_size = word2vec.vectors.shape[0]
    args.embed_size = word2vec.vectors.shape[1]
    embeddings = np.zeros((vocab_size+1, args.embed_size), dtype="float32")
    embeddings[:vocab_size] = word2vec.vectors
    
    if args.no_pretrain:
        embeddings = None
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_file = os.path.join(args.log_dir, '%d.log'%time.time())
    
    model = CodeModel(args, vocab_size+1, label_size=104, log_file=log_file, pretrain_emb=embeddings)
    if not args.only_test:
        model.train(train_data, valid_data, test_data)

    model_path = os.path.join(args.model_dir, 'best.model')
    model.load_model(model_path)
    test_acc = model.evaluate(test_data)
    print("Using model %s, Test Acc: %.4f" % (model_path, test_acc))

    

if __name__ == '__main__':
    main()