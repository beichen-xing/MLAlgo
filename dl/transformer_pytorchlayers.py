import re
from collections import Counter

import torch
import torch.nn as nn
import math


from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# class Transformer(nn.Module):
#
#     def __init__(self,
#                  src_vocab_size,
#                  tgt_vocab_size,
#                  embed_size,
#                  num_heads,
#                  num_encoder_layers,
#                  num_decoder_layers,
#                  forward_expansion,
#                  dropout,
#                  max_length,
#                  device):
#         super(Transformer, self).__init__()
#         self.device = device
#
#         self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
#         self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
#         self.positional_encoding = nn.Embedding(max_length, embed_size)
#
#         self.transformer = nn.Transformer(
#             d_model=embed_size,
#             nhead=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=forward_expansion * embed_size,
#             dropout=dropout,
#             batch_first=True
#         )
#
#         self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
#
#     def forward(self, src, tgt,
#                 src_mask=None,
#                 tgt_mask=None,
#                 src_padding_mask=None,
#                 tgt_padding_mask=None,
#                 memory_key_padding_mask=None):
#         batch_size, src_seq_length = src.shape
#         batch_size, tgt_seq_length = tgt.shape
#
#         src_positions = (
#             torch.arange(0, src_seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
#         )
#         tgt_positions = (
#             torch.arange(0, tgt_seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
#         )
#
#         src_embed = self.src_embedding(src) + self.positional_encoding(src_positions)
#         tgt_embed = self.tgt_embedding(tgt) + self.positional_encoding(tgt_positions)
#
#         output = self.transformer(
#             src_embed,
#             tgt_embed,
#             src_mask,
#             tgt_mask,
#             src_padding_mask,
#             tgt_padding_mask,
#             memory_key_padding_mask
#         )
#
#         return self.fc_out(output)


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_size,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 max_length,
                 device):
        super(Transformer, self).__init__()
        self.device = device

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def forward(self, src, tgt,
                src_mask=None,
                tgt_mask=None,
                src_padding_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=None):
        batch_size, src_seq_length = src.shape
        batch_size, tgt_seq_length = tgt.shape

        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        )
        tgt_positions = (
            torch.arange(0, tgt_seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        )

        src_embed = self.src_embedding(src) + self.positional_encoding(src_positions)
        tgt_embed = self.tgt_embedding(tgt) + self.positional_encoding(tgt_positions)

        output = self.transformer(
            src_embed,
            tgt_embed,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        return self.fc_out(output)



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

model = Transformer(len(src_vocab), len(tgt_vocab), EMBED_SIZE, NUM_HEADS, NUM_LAYERS, NUM_LAYERS, FF_DIM // EMBED_SIZE, DROPOUT, MAX_LEN, DEVICE).to(DEVICE)
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