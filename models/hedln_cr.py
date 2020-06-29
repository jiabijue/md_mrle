import torch
import torch.nn as nn


class HEDLN_CR(nn.Module):
    def __init__(self, n_filters, kernel_sizes, strides, dropout,
                 rnn_hid_size, rnn_n_layers, bidirectional,
                 num_classes1=2, num_classes2=3):
        super(HEDLN_CR, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_directions = 2 if bidirectional else 1

        in_channels = [1] + n_filters[:-1]

        # net1
        self.cnn1 = []
        for i in range(len(n_filters)):
            self.cnn1 += [nn.Conv1d(in_channels[i], n_filters[i], kernel_sizes[i], strides[i]),
                          nn.BatchNorm1d(n_filters[i]),
                          nn.ReLU(inplace=False),
                          nn.Dropout(dropout),
                          nn.MaxPool1d(kernel_sizes[i], strides[i])]
        self.cnn1 = nn.Sequential(*self.cnn1)

        self.rnn1 = nn.GRU(n_filters[-1], rnn_hid_size, rnn_n_layers,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc1 = nn.Linear(self.num_directions * rnn_hid_size, num_classes1)
        self.act1 = nn.Sigmoid()

        # net2
        self.cnn2 = []
        for i in range(len(n_filters)):
            self.cnn2 += [nn.Conv1d(in_channels[i], n_filters[i], kernel_sizes[i], strides[i]),
                          nn.BatchNorm1d(n_filters[i]),
                          nn.ReLU(inplace=False),
                          nn.Dropout(dropout),
                          nn.MaxPool1d(kernel_sizes[i], strides[i])]
        self.cnn2 = nn.Sequential(*self.cnn2)

        self.rnn2 = nn.GRU(n_filters[-1]*2, rnn_hid_size, rnn_n_layers,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc2 = nn.Linear(self.num_directions * rnn_hid_size, num_classes2)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        """ batch-first
        x:       (seq_len, bz, input_size)
        rnn_out1: (seq_len, bz, num_directions * hidden_size)
        rnn_out2: (seq_len, bz, num_directions * hidden_size)
        """
        T, B, S1 = x.shape

        cnn_in1 = x.reshape(T * B, 1, S1)
        cnn_out1 = self.cnn1(cnn_in1)  # (T*B, 256, 1)
        rnn_in1 = cnn_out1.reshape(T, B, -1)
        rnn_out1, _ = self.rnn1(rnn_in1)
        fc_in1 = rnn_out1.reshape(T * B, self.num_directions * self.rnn_hid_size)
        fc_out1 = self.fc1(fc_in1)
        logits1 = self.act1(fc_out1)
        logits1 = logits1.reshape(T, B, self.num_classes1)

        cnn_in2 = torch.cat((x, rnn_out1), 2).reshape(T * B, 1, -1)
        cnn_out2 = self.cnn2(cnn_in2)  # (T*B, 256, 1)
        rnn_in2 = cnn_out2.reshape(T, B, -1)
        rnn_out2, _ = self.rnn2(rnn_in2)
        fc_in2 = rnn_out2.reshape(T * B, self.num_directions * self.rnn_hid_size)
        fc_out2 = self.fc2(fc_in2)
        logits2 = self.act2(fc_out2)
        logits2 = logits2.reshape(T, B, self.num_classes2)

        return logits1, logits2
