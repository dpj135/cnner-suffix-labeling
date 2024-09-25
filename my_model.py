import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs

class MultiHeadBiaffine(nn.Module):
    def __init__(self, input_dim, hidden_dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert hidden_dim%n_head==0
        in_head_dim = hidden_dim//n_head
        out = hidden_dim if out is None else out
        assert out%n_head == 0
        out_head_dim = out//n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """

        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        h=self.head_mlp(x)
        v=self.tail_mlp(x)
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        w = w.permute(0,2,3,1)
        return w

class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x



class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                bias=False, groups=groups)

    def forward(self, x):
        """

        :param x:
        :param mask:
        :return:
        """
        #x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x):
        x=x.permute(0,3,1,2)
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x)
            else:
                x = layer(x)
        return _x.permute(0,2,3,1)
class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.4):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        #x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        self.mh_biaffine = MultiHeadBiaffine(config.lstm_hid_size,config.biaffine_size)

        conv_input_size = config.lstm_hid_size + config.dist_emb_size
        #self.convLayer = ConvolutionLayer(conv_input_size, conv_input_size//2, config.dilation, config.conv_dropout)
        self.convLayer=MaskCNN(conv_input_size, conv_input_size)
        self.dropout = nn.Dropout(config.emb_dropout)

        self.mlp=MLP(conv_input_size , config.label_num)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        span_reps = self.mh_biaffine(word_reps)
        span_length_emb = self.dis_embs(dist_inputs)
        #print(span_reps.shape)
        #print(span_length_emb.shape)
        #sys.exit()
        conv_inputs = torch.cat([span_reps,span_length_emb], dim = -1)

        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        #注意shape
        #print(conv_inputs.shape,conv_outputs.shape)
        outputs=self.mlp(conv_inputs+conv_outputs)

        return outputs
