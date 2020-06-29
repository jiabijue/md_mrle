import torch.nn as nn


class CLDNN(nn.Module):

    def __init__(self, n_filters, kernel_sizes, strides, dropout,
                 rnn_hid_size, rnn_n_layers, bidirectional, num_classes=3):
        super(CLDNN, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.num_classes = num_classes
        self.num_directions = 2 if bidirectional else 1

        self.cnn = []
        in_channels = [1] + n_filters[:-1]
        for i in range(len(n_filters)):
            self.cnn += [nn.Conv1d(in_channels[i], n_filters[i], kernel_sizes[i], strides[i]),
                         nn.BatchNorm1d(n_filters[i]),
                         nn.ReLU(inplace=False),
                         nn.Dropout(dropout),
                         nn.MaxPool1d(kernel_sizes[i], strides[i])]
        self.cnn = nn.Sequential(*self.cnn)

        self.rnn = nn.GRU(n_filters[-1], rnn_hid_size, rnn_n_layers,
                          dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.num_directions * rnn_hid_size, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """ batch-first
        x:       (seq_len, bz, input_size)
        """
        T, B, S1 = x.shape
        x = x.reshape(T * B, 1, S1)
        cnn_out = self.cnn(x)  # (T*B, 256, 1)

        rnn_in = cnn_out.reshape(T, B, -1)
        rnn_out, _ = self.rnn(rnn_in)

        fc_in = rnn_out.reshape(T * B, self.num_directions * self.rnn_hid_size)
        fc_out = self.fc(fc_in)
        logits = self.act(fc_out)
        logits = logits.reshape(T, B, self.num_classes)

        return logits
