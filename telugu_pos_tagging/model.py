import torch
from torch import nn
import torch.nn.functional as F
import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMTagger(nn.Module):
    def __init__(self,vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = 200
        self.num_layers = 1
        self.embedding_dim = 200
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim = self.embedding_dim).to(device)
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            dropout = 0.2,
            batch_first = True,
        )
        self.hidden2tag = nn.Linear(3*self.hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, hidden = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out.reshape(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 10, self.hidden_dim).to(device),
                    torch.zeros(self.num_layers, 10,self.hidden_dim).to(device))
