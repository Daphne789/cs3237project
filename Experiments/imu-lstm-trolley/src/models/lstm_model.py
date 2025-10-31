import torch
import torch.nn as nn

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=None, dropout=0.3, bidirectional=True, regression_dim=0):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        h_out = hidden_size * (2 if bidirectional else 1)
        # a small MLP head for classification/regression
        self.fc_shared = nn.Sequential(
            nn.Linear(h_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = None
        self.regressor = None
        if num_classes is not None:
            self.classifier = nn.Linear(128, num_classes)
        if regression_dim and regression_dim > 0:
            self.regressor = nn.Linear(128, regression_dim)

    def forward(self, x):
        # x: (batch, seq, feat)
        out, _ = self.lstm(x)  # (batch, seq, hidden*directions)
        # use mean pooling over time (more stable than last timestep)
        pooled = out.mean(dim=1)
        shared = self.fc_shared(pooled)
        out_clf = self.classifier(shared) if self.classifier is not None else None
        out_reg = self.regressor(shared) if self.regressor is not None else None
        return out_clf, out_reg
    