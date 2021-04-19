import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from utils import batch_generator


class RnnModel(nn.Module):
    def __init__(self, vocab_size, label_size, emb_dim, rnn_state_dim, 
                    pretrain_emb=None, fix_emb=False, device='cpu'):
        super(RnnModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if pretrain_emb is not None:
            self.embedding = self.embedding.from_pretrained(pretrain_emb, freeze=fix_emb)       # [8307, 128]
        
        self.pre_gru = nn.GRU(emb_dim, rnn_state_dim, bidirectional=True, batch_first=True)
        self.post_gru = nn.GRU(emb_dim, rnn_state_dim, bidirectional=True, batch_first=True)
        self.level_gru = nn.GRU(emb_dim, rnn_state_dim, bidirectional=True, batch_first=True)

        self.pre_linear1 = nn.Linear(rnn_state_dim*6, rnn_state_dim*2)
        self.pre_linear2 = nn.Linear(rnn_state_dim*2, label_size)
        self.activation = torch.relu

    def forward(self, inputs):
        """
            inputs: [pre_order_seq, post_order_seq, level_order_seq]

            pre_order_seq: list with length=batch_size, [LongTensor, LongTensor, ...],
                            where LongTensor has shape [seq_len,]
        """
        rnns = [self.pre_gru, self.post_gru, self.level_gru]
        outputs = []
        for i, inp in enumerate(inputs):
            real_len = [t.shape[0] for t in inp]
            inp = rnn_utils.pad_sequence(inp, batch_first=True, padding_value=0) # [batch_size, batch_max_seq_len]
            embs = self.embedding(inp)
            pack_inp = rnn_utils.pack_padded_sequence(embs, lengths=real_len, batch_first=True, enforce_sorted=False)

            out, hidden = rnns[i](pack_inp)
            # out, hidden = self.pre_gru(pack_inp)
            out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True, padding_value=float('-inf'))
            out = torch.transpose(out, 1, 2)
            out = F.max_pool1d(out, out.size(2)).squeeze(2)     # [batch_size, rnn_state_dim*2]
            outputs.append(out)

        outputs = torch.cat(outputs, dim=-1)
        h = self.activation(self.pre_linear1(outputs))
        return self.pre_linear2(h)


class CodeModel():
    def __init__(self, args, vocab_size, label_size, log_file='./rst.log', pretrain_emb=None):
        if torch.cuda.is_available() and args.gpu > -1:
            self.device = torch.device('cuda:%d'%args.gpu)
            print("Using GPU%d to train" % args.gpu)
        else:
            self.device = torch.device('cpu')
            print("Using CPU to train")
        
        # write train/valid/test result into file
        self.log = log_file
        with open(self.log, 'a') as f:
            f.write(str(args)+'\n')
        self.model_dir = args.model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.args = args
        self.model = RnnModel(vocab_size, label_size, args.embed_size, args.rnn_state_dim, pretrain_emb, args.fix_emb, self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.optimizer = torch.optim.Adamax(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_data, valid_data, test_data):
        eval_freq = self.args.eval_freq
        bs = self.args.batch_size
        EPOCHS = self.args.epochs

        best_valid_acc = 0
        best_valid_model = 0
        for epoch in tqdm(range(EPOCHS)):
            train_loss = 0.
            train_acc = 0.
            total = 0.
            train_gen = batch_generator(train_data, bs, shuffle=True, max_len=self.args.max_seq_len, device=self.device)
            for batch in train_gen:
                batch_label = batch[3]
                self.model.train()
                self.optimizer.zero_grad()
                batch_logit = self.model(batch[:3])
                loss = self.criterion(batch_logit, batch_label)
                loss.backward()
                self.optimizer.step()

                # cal train acc
                _, batch_pred = torch.max(batch_logit, 1)
                train_acc += (batch_pred==batch_label).sum()
                train_loss += (loss.item() * batch_label.shape[0])
                total += batch_label.shape[0]
            train_loss /= total
            train_acc /= total
            train_record = '[Epoch: %3d/%3d] Training Loss: %.4f, Training Acc: %.4f' \
                                 % (epoch, EPOCHS, train_loss, train_acc)
            print(train_record)
            with open(self.log, 'a') as f:
                f.write(train_record+'\n')

            if eval_freq > 0 and epoch % eval_freq == 0:
                valid_acc = self.evaluate(valid_data)
                test_acc = self.evaluate(test_data)
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    best_valid_model = epoch
                    self.save_model()
                eval_record = '[Epoch: %3d/%3d] Valid ACC: %.4f, Test Acc: %.4f, Best Epoch: %d' % \
                        (epoch, EPOCHS, valid_acc, test_acc, best_valid_model)
                print(eval_record)
                with open(self.log, 'a') as f:
                    f.write(eval_record+'\n')



    def evaluate(self, data):
        data_gen = batch_generator(data, self.args.batch_size, max_len=self.args.max_seq_len, device=self.device)
        eval_acc = 0.
        total = 0.
        for batch in data_gen:
            batch_label = batch[3]
            self.model.eval()
            with torch.no_grad():
                batch_logit = self.model(batch[:3])
            
            _, batch_pred = torch.max(batch_logit, 1)
            eval_acc += (batch_pred==batch_label).sum()
            total += batch_label.shape[0]
        eval_acc /= total
        return eval_acc

    
    def save_model(self):
        """
            Saves the model into model_dir 
        """
        save_model_path = os.path.join(self.model_dir, 'best.model')
        torch.save(self.model.state_dict(), save_model_path)

    def load_model(self, model_path):
        """
            Restores the model from model_dir
        """
        assert os.path.exists(model_path), "%s not exists" % model_path
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        print('Model loaded from %s' % model_path)
