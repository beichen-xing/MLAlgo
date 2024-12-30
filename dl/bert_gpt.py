import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, dropout=0.1):
        super(BERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
             for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, position_ids, attention_mask):

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        print(x.shape, attention_mask.shape)
        x = self.layer_norm(x)
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * -1e9
        # attention_mask = attention_mask.squeeze().bool()

        for layer in self.layers:
            print(x.shape, attention_mask.shape)
            x = layer(x, src_key_padding_mask=attention_mask)

        logits_mlm = self.mlm_head(x)
        cls_output = x[:, 0, :]
        logits_nsp = self.nsp_head(cls_output)

        return logits_mlm, logits_nsp


class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, dropout=0.1):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
             for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, position_ids, attention_mask):
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.layer_norm(x)
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * -1e9
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1).bool()
        print(attention_mask.shape, causal_mask.shape)
        memory = torch.zeros(x.size(0), 1, x.size(-1), device=x.device)

        for layer in self.layers:
            x = layer(x, tgt_mask=causal_mask, memory=memory)

        logits = self.head(x)
        return logits


class SimpleDataset(Dataset):
    def __init__(self, vocab_size, seq_len, dataset_size):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(device)

        optimizer.zero_grad()
        # logits_mlm, _ = model(input_ids, position_ids, attention_mask)
        # loss = criterion(logits_mlm.view(-1, logits_mlm.size(-1)), labels.view(-1))

        logits = model(input_ids, position_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def inference(model, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(device)
        logits_mlm, logits_nsp = model(input_ids.to(device), position_ids, attention_mask.to(device))
        # logits = model(input_ids.to(device), position_ids, attention_mask.to(device))

    return logits_mlm, logits_nsp
    # return logits


vocab_size = 30522
hidden_size = 768
num_layers = 12
num_heads = 12
max_seq_len = 128
batch_size = 8
epochs = 2
dataset_size = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SimpleDataset(vocab_size, max_seq_len, dataset_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

bert_model = BERT(vocab_size, hidden_size, num_layers, num_heads, max_seq_len).to(device)
gpt_model = GPT(vocab_size, hidden_size, num_layers, num_heads, max_seq_len).to(device)

optimizer_bert = optim.AdamW(bert_model.parameters(), lr=5e-5)
optimizer_gpt = optim.AdamW(gpt_model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Total parameters in BERT:", count_parameters(bert_model))
print("Total parameters in GPT:", count_parameters(gpt_model))


# print(f"Training BERT... {time.localtime()}")
# for epoch in range(epochs):
#     loss = train(bert_model, dataloader, optimizer_bert, criterion, device)
#     print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, {time.localtime()}")

# print(f"Training GPT... {time.localtime()}")
# for epoch in range(epochs):
#     loss = train(gpt_model, dataloader, optimizer_gpt, criterion, device)
#     print(f"Epoch {epoch + 1}, Loss: {loss:.4f} {time.localtime()}")

print(f"Running inference...{time.localtime()}")
sample_input_ids = torch.randint(0, vocab_size, (1, max_seq_len))
sample_attention_mask = torch.ones(1, max_seq_len)
# sample_attention_mask = sample_attention_mask.unsqueeze(1).unsqueeze(2)

# bert_logits_mlm, bert_logits_nsp = inference(bert_model, sample_input_ids, sample_attention_mask, device)
# print("BERT MLM logits shape:", bert_logits_mlm.shape)
# print("BERT NSP logits shape:", bert_logits_nsp.shape)

gpt_logits = inference(gpt_model, sample_input_ids, sample_attention_mask, device)
print("GPT logits shape:", gpt_logits.shape)



