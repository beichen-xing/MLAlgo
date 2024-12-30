import re
from collections import Counter

import torch
import torch.nn as nn
import math


from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    # embed_size: word length, max_len: word count
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask=None):
        N, query_len, embed_size = queries.shape
        _, key_len, _ = keys.shape

        # Query: Defines what the model is looking for.
        # Key: Provides descriptors of elements.
        # Value: Contains the actual information used in the output.

        Q = self.query(queries).view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(keys).view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(values).view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nhqd,nhkd->nhqk", Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            energy.masked_fill_(mask == 0, float('-inf'))

        # Softmax is applied across the key_len dimension
        # because attention weights are calculated for each query over all keys
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nhqk,nhvd->nhqd", attention, V)
        out = out.transpose(1, 2).contiguous().view(N, query_len, embed_size)

        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # self.attention = MultiHeadAttention(embed_size, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_dim)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # attention = self.attention(x, x, x, mask)
        attention, _ = self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attention))
        feedforward = self.ff(x)
        x = self.norm2(x + self.dropout(feedforward))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        # self.cross_attention = MultiHeadAttention(embed_size, num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                                     dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_dim)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # self_attn = self.self_attention(x, x, x, tgt_mask)
        self_attn, _ = self.self_attention(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(self_attn))

        # cross_attn = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        cross_attn, _ = self.cross_attention(x, encoder_out, encoder_out, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        feedforward = self.ff(x)
        x = self.norm3(x + self.dropout(feedforward))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, num_layers, src_vocab_size, tgt_vocab_size, max_len, dropout):
        super(Transformer, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoing = PositionalEncoding(embed_size, max_len)

        self.encoder = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def make_src_mask(self, src):
        # return (src != 0).unsqueeze(1).unsqueeze(2)
        return (src != 0)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        # return torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len).to(tgt.device)
        return torch.tril(torch.ones(tgt_len, tgt_len)).bool()

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.src_embed(src)
        enc_out = self.positional_encoing(enc_out)

        for layer in self.encoder:
            enc_out = layer(enc_out, src_key_padding_mask=src_mask)

        dec_out = self.tgt_embed(tgt)
        dec_out = self.positional_encoing(dec_out)
        tgt_padding_mask = self.make_src_mask(tgt)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_mask)

        out = self.fc_out(dec_out)
        return out


def custom_tokenizer(sentence):
    sentence = re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence.split()


def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), start=4)}
    vocab.update({"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3})
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab


class TranslationDataSet(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = [self.src_vocab.get("<sos>", 1)] \
                     + [self.src_vocab.get(token, 3) for token in self.src_tokenizer(src)] \
                     + [self.src_vocab.get("<eos>", 2)]
        tgt_tokens = [self.tgt_vocab.get("<sos>", 1)] \
                     + [self.tgt_vocab.get(token, 3) for token in self.tgt_tokenizer(tgt)] \
                     + [self.tgt_vocab.get("<eos>", 2)]

        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]
        src_tokens += [self.src_vocab.get("<pad>", 0)] * (self.max_len - len(src_tokens))
        tgt_tokens += [self.tgt_vocab.get("<pad>", 0)] * (self.max_len - len(tgt_tokens))

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


data = [
    ("I love programming", "J'aime programmer"),
    ("How are you?", "Comment ça va?"),
    ("Good morning", "Bonjour"),
    ("See you later", "À plus tard"),
    ("Thank you", "Merci"),
    ("What is your name?", "Comment vous appelez-vous?"),
]

MAX_LEN = 50
BATCH_SIZE = 32
EMBED_SIZE = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 6
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

src_vocab, src_reverse_vocab = build_vocab([src for src, tgt in data], custom_tokenizer)
tgt_vocab, tgt_reverse_vocab = build_vocab([tgt for src, tgt in data], custom_tokenizer)
dataset = TranslationDataSet(data, src_vocab, tgt_vocab, custom_tokenizer, custom_tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Transformer(EMBED_SIZE, NUM_HEADS, FF_DIM, NUM_LAYERS, len(src_vocab), len(tgt_vocab), MAX_LEN, DROPOUT).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(dataloader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()
        output = model(src, tgt_input)
        output = output.view(-1, output.size(-1))

        loss = loss_fn(output, tgt_output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")


def translate_sentence(model, sentence, src_vocab, tgt_vocab, tgt_reverse_vocab, src_tokenizer, max_len, device):
    model.eval()
    tokens = [src_vocab["<sos>"]] + [src_vocab.get(token, 3) for token in src_tokenizer(sentence)] + [src_vocab["<eos>"]]
    tokens = tokens[:max_len] + [src_vocab["<pad>"]] * (max_len - len(tokens))
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    tgt_tokens = [tgt_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
            nxt_token = output.argmax(-1)[:, -1].item()
            tgt_tokens.append(nxt_token)
            if nxt_token == tgt_vocab["<eos>"]:
                break
    return " ".join([tgt_reverse_vocab[token] for token in tgt_tokens[1:-1]])


test_sentence = "Good morning"
translation = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, tgt_reverse_vocab, custom_tokenizer, MAX_LEN, DEVICE)
print(f"Translation: {translation}")