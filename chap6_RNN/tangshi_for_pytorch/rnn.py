import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__  # 获取类名
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("初始化 线性权重 ")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))

    def forward(self, input_sentence):
        """
        :param input_sentence: 输入的句子张量，包含多个单词的索引
        :return: 返回单词的嵌入张量
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        #########################################
        # 定义 LSTM，输入大小为 embedding_dim，输出大小为 lstm_hidden_dim
        # LSTM应有两层，输入输出形状为 (batch, seq, feature)
        self.rnn_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True)

        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)  # 用于将LSTM的输出映射到词汇表大小
        self.apply(weights_init)  # 调用权重初始化函数。

        self.softmax = nn.LogSoftmax(dim=-1)  # 激活函数为 LogSoftmax
        # self.tanh = nn.Tanh()

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)  # 获取词嵌入
        # print(batch_input.size()) # 输出输入的大小

        ################################################
        # 将 "batch_input" 输入到 self.rnn_lstm 中
        # 隐藏输出命名为 output，初始的隐藏状态和细胞状态都设置为零。
        h0 = torch.zeros(2, 1, self.lstm_dim).to(sentence.device)  # 初始化隐藏状态，2是LSTM的层数
        c0 = torch.zeros(2, 1, self.lstm_dim).to(sentence.device)  # 初始化细胞状态

        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))  # 获取LSTM的输出
        ################################################

        out = output.contiguous().view(-1, self.lstm_dim)  # 将输出展平成一个长向量

        out = F.relu(self.fc(out))  # 全连接层，ReLU激活

        out = self.softmax(out)  # 使用Softmax

        if is_test:
            prediction = out[-1, :].view(1, -1)  # 预测最后一个词
            output = prediction
        else:
            output = out

        # print(out)
        return output
