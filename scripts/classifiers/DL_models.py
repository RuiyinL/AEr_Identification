import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TextCNN classifier Parameter
def parse_args():
    parser = argparse.ArgumentParser(description='TextCNN text classifier')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epoch', type=int, default=200, help='number of training epochs')
    parser.add_argument('-out_channels', type=int, default=100, help='number of input channels')
    parser.add_argument('-filter_sizes', type=str, default='3,4,5', help='filter sizes')
    parser.add_argument('-embedding_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('-max_length', type=int, default=2000, help='max length of a sentence')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-label_num', type=int, default=2, help='number of labels')
    # parser.add_argument('-static', type=bool, default=True, help='whether to use pre-trained word embedding')
    parser.add_argument('-fine_tune', type=bool, default=True, help='whether to fine-tune the encoder')
    # parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-checkpoint_interval', type=int, default=1, help='Run validation and save model parameters at this interval')
    parser.add_argument('-test_interval', type=int, default=100,help='how many epochs to wait before testing')
    parser.add_argument('-early_stopping', type=int, default=500, help='Early Stopping after n epoch')
    parser.add_argument('-save_best', type=bool, default=True, help='save the best model')
    parser.add_argument('-save_dir', type=str, default='checkpoints', help='directory to save models')
    parser.add_argument('--label_path0', type=str, default='~/experiment/data/Randomly_selected_comments.xlsx')
    parser.add_argument('--label_path1', type=str, default='~/experiment/data/Violation symptoms.xlsx')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.25)
    parser.add_argument('--random_seed', type=int, default=6)
    parser.add_argument('--n_per_class', type=int, default=599)
    parser.add_argument('--test_path', type=str, default='~/experiment/data/Test set.xlsx')
    parser.add_argument('--test_sheet', type=str, default='combination')
    parser.add_argument("-do_train", default=True, help="Whether to start training.")
    parser.add_argument("-do_test", default=True, help="Whether to start testing.")
    parser.add_argument("-TextCNN_voc", default=False, help="Whether to load TextCNN_voc model")
    parser.add_argument("-TextCNN_w2v", default=False, help="Whether to load TextCNN_w2v model")
    parser.add_argument("-TextCNN_fastText", default=False, help="Whether to load TextCNN_fastText model")
    parser.add_argument("-TextCNN_GloVe", default=True, help="Whether to load TextCNN_GloVe model")
    # parser.add_argument("-SO_w2v_200", default=True, help="Whether to load SO_w2v_200 pre-trained vectors")
    # parser.add_argument("-FastText_200", default=False, help="Whether to load FastText_200 pre-trained vectors")
    args = parser.parse_args()
    return args


class TextCNN_voc(nn.Module):
    def __init__(self, args, vocab_size):
        super(TextCNN_voc, self).__init__()
        '''Define parameters'''
        self.args = args
        label_num = args.label_num
        out_channels = args.out_channels
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        # vocab_size = args.vocab_size
        self.vocab_size = vocab_size
        embedding_dim = args.embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes) * out_channels, label_num)

    def forward(self, x):
        x = x.long().to(DEVICE)
        x = self.embedding(x)
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        model_out = self.linear(x)
        return model_out


    def get_embedding(self, x):
        return self.embedding(torch.LongTensor(np.array(x)))


class TextCNN_w2v(nn.Module):
    def __init__(self, args):
        super(TextCNN_w2v, self).__init__()
        '''Define parameters'''
        self.args = args
        label_num = args.label_num
        out_channels = args.out_channels
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        embedding_dim = args.embedding_dim
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes) * out_channels, label_num)

    def forward(self, x):
        """Perform a forward pass through the network.
        x:  A tensor of token ids with shape (batch_size, max_sent_length)
        """
        x = x.to(DEVICE)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        model_out = self.linear(x)
        return model_out


class TextCNN_fastText(nn.Module):
    def __init__(self, args):
        super(TextCNN_fastText, self).__init__()
        '''Define parameters'''
        self.args = args
        label_num = args.label_num
        out_channels = args.out_channels
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        embedding_dim = args.embedding_dim
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes) * out_channels, label_num)

    def forward(self, x):
        x = x.to(DEVICE)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        model_out = self.linear(x)
        return model_out

class TextCNN_GloVe(nn.Module):
    def __init__(self, args):
        super(TextCNN_GloVe, self).__init__()
        '''Define parameters'''
        self.args = args
        label_num = args.label_num
        out_channels = args.out_channels
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        embedding_dim = args.embedding_dim
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes) * out_channels, label_num)

    def forward(self, x):
        x = x.to(DEVICE)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        model_out = self.linear(x)
        return model_out
