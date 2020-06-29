import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout, bidirectional, num_classes=3):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                           dropout=dropout, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.out_layer = nn.Sequential(
            nn.Linear(num_directions * hidden_size, num_classes),
            nn.Sigmoid()
        )
        self.num_classes = num_classes

    def forward(self, x):
        """ batch-first
        x:       (seq_len, bz, input_size)
        rnn_out: (seq_len, bz, num_directions * hidden_size)
        logits:  (seq_len, bz, num_classes)
        """
        T, B, S1 = x.shape
        rnn_out, _ = self.rnn(x)

        S2 = rnn_out.shape[-1]
        rnn_out = rnn_out.reshape(-1, S2)
        output = self.out_layer(rnn_out)

        logits = output.reshape(T, B, self.num_classes)
        return logits
