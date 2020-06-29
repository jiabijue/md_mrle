import torch
import torch.nn as nn


class HEDLN_R(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout, bidirectional, num_classes1, num_classes2):
        super(HEDLN_R, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2

        self.rnn1 = nn.GRU(input_size, hidden_size, num_layers,
                           dropout=dropout, bidirectional=bidirectional)

        self.fc1 = nn.Linear(self.num_directions * hidden_size, num_classes1)
        self.act1 = nn.Sigmoid()  # sigmoid+bce

        self.rnn2 = nn.GRU(input_size + self.num_directions * hidden_size,
                           hidden_size, num_layers,
                           dropout=dropout, bidirectional=bidirectional)
        self.fc2 = nn.Linear(self.num_directions * hidden_size, num_classes2)
        self.act2 = nn.Sigmoid()  # sigmoid+bce

    def forward(self, x):
        """ batch-first
        x:        (seq_len, bz, input_size)
        rnn1_out: (seq_len, bz, num_directions * hidden_size)
        rnn2_out: (seq_len, bz, num_directions * hidden_size)
        logits1:  (seq_len*bz, 2)
        logits2:  (seq_len*bz, 3)
        """
        T, B, S1 = x.shape
        rnn1_out, _ = self.rnn1(x)

        S2 = rnn1_out.shape[-1]
        fc1_in = rnn1_out.reshape(-1, S2)
        logits1 = self.fc1(fc1_in)
        logits1 = self.act1(logits1)  # sigmoid+bce
        logits1 = logits1.reshape(T, B, self.num_classes1)  # sigmoid+bce

        rnn2_in = torch.cat((x, rnn1_out), 2)
        rnn2_out, _ = self.rnn2(rnn2_in)

        fc2_in = rnn2_out.reshape(-1, S2)
        logits2 = self.fc2(fc2_in)
        logits2 = self.act2(logits2)  # sigmoid+bce
        logits2 = logits2.reshape(T, B, self.num_classes2)  # sigmoid+bce

        return logits1, logits2
