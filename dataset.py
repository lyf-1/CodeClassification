import os
import pandas as pd
from utils import check_path


class Dataset():
    def __init__(self, train_file, valid_file, test_file):
        print("Beinging preprocess data...")
        self.set_types = ['train', 'valid', 'test']
        self.files = [train_file, valid_file, test_file]
        for f in self.files:
            assert check_path(f), "%s not exists" % f
        self.save_dir = os.path.dirname(train_file)
    
    def process(self):
        print("Convert C code to AST, and preorder/postorder/levelorder traverse for AST")
        for i, f in enumerate(self.files):
            self._c2ast(f, self.set_types[i])
        
        print("Training Word2Vec model")
        self.train_word2vec(size=128, augmentation=True)

        print("Building index sequence")
        for i, f in enumerate(self.files):
            self.build_seq_index(self.set_types[i], augmentation=True)
    

    def _c2ast(self, fpath, set_name):
        save_path = os.path.join(self.save_dir, "%s_ast.pkl"%set_name)
        if check_path(save_path):
            return
        
        from pycparser import c_parser
        from parsetree import TreeTraverse
        cparser = c_parser.CParser()
        tparser = TreeTraverse()
        df = pd.read_pickle(fpath)
        df['ast'] = df['code'].apply(cparser.parse)
        df['pre'] = df['ast'].apply(tparser.preorder)
        df['post'] = df['ast'].apply(tparser.postorder)
        df['level'] = df['ast'].apply(tparser.levelorder)
        df.to_pickle(save_path)
        
    
    def train_word2vec(self, input_file=None, size=128, augmentation=True):
        self.size = size
        save_emb_dir = os.path.join(self.save_dir, 'embedding')
        if not check_path(save_emb_dir):
            os.makedirs(save_emb_dir)
        save_emb_file = os.path.join(save_emb_dir, 'node_w2v_%d'%size)
        if check_path(save_emb_file):
            print("pretrained embedding already exisits")
            return

        if input_file is None or not check_path(input_file):
            input_file = os.path.join(self.save_dir, 'train_ast.pkl')
        trees = pd.read_pickle(input_file)

        print("Original sample number:", len(trees))
        if augmentation:
            cols = ['pre', 'post', 'level']
        else:
            cols = ['pre']
        corpus = []
        for c in cols:
            corpus += trees[c].tolist()
        print("Sample number to train Word2Vec model:", len(corpus))

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(save_emb_file)


    def build_seq_index(self, set_type, augmentation=True):
        save_path = os.path.join(self.save_dir, 'embedding', '%s.pkl'%set_type)
        if check_path(save_path):
            print("id sequence file already exists")
            return

        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(os.path.join(self.save_dir, 'embedding', 'node_w2v_%d'%self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(seq_list):
            result = [vocab[w].index if w in vocab else max_token for w in seq_list]
            return result

        if augmentation:
            cols = ['pre', 'post', 'level']
        else:
            cols = ['pre']
        fpath = os.path.join(self.save_dir, '%s_ast.pkl'%set_type)
        trees = pd.read_pickle(fpath)
        for c in cols:
            trees[c] = trees[c].apply(tree_to_index)
        cols = cols+['label']
        trees[cols].to_pickle(save_path)

        seq_len = trees['pre'].apply(len)
        print("min/max sequence lenght: ", seq_len.min(), seq_len.max())


if __name__ == '__main__':
    data = Dataset(train_file='./data/train.pd', valid_file='./data/valid.pd', test_file='./data/test.pd')
    data.process()


    # dd = {'a':[[1,2,3],[1],[2,3]], 'b':[[4],[5],[4,5]], 'c':[[7,8],[9],[10,101]]}
    # df = pd.DataFrame(dd)
    # print(df)

    # def to_index(seq):
    #     return [i for i, w in enumerate(seq)]
    
    # for col in ['a','b']:
    #     df[col] = df[col].apply(to_index)
    # print()
    # print(df)

    # # df1 = df['a']
    # # print()
    # # print(df1)
    # # print(df1.tolist())