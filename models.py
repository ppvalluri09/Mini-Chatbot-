import torch

class ChatBot(torch.nn.Module):
    def __init__(self, embedding_weights, out_features):
        super(ChatBot, self).__init__()
        self.emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))
        self.emb.requires_grad = False
        self.lstm = torch.nn.LSTM(embedding_weights.shape[1], hidden_size=128)
        self.fc = torch.nn.Linear(64*128, out_features)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        emb_out = self.dropout(self.emb(x))
        lstm_out, _ = self.lstm(emb_out)
        return self.softmax(self.fc(lstm_out.view(x.size(0), -1)))