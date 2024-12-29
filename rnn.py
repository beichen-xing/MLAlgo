import numpy as np
import numpy.random
import torch
import torch.nn as nn
import torch.optim as optim


sequence_length = 10
num_sequences = 1000

def create_dataset(num_sequences, sequence_length):
    X = []
    y = []

    for _ in range(num_sequences):
        sequence = torch.rand(sequence_length)
        X.append(sequence[:-1])
        y.append(sequence[-1])

    return torch.stack(X), torch.stack(y)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # self.weights = {
        #     "Wxh": np.random.randn(input_dim, hidden_units),
        #     "Whh": np.random.randn(hidden_units, hidden_units),
        #     "Why": np.random.randn(hidden_units, output_dim)
        # }
        # self.biases = {
        #     "bh": np.zeros((1, hidden_units)),
        #     "by": np.zeros((1, output_dim))
        # }
        # self.Wxh = nn.Parameter(torch.randn(input_dim, hidden_units) * 0.01)
        # self.Whh = nn.Parameter(torch.randn(hidden_units, hidden_units) * 0.01)
        # self.Why = nn.Parameter(torch.randn(hidden_units, output_dim) * 0.01)
        # self.bh = nn.Parameter(torch.zeros(hidden_units))
        # self.by = nn.Parameter(torch.zeros(output_dim))
        self.rnn = nn.RNN(input_dim, hidden_units, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def tanh(self, x):
        return torch.tanh(x)

    def forward(self, x):
        batch_size, time_steps, _ = x.shape
        h = torch.zeros((2, batch_size, self.hidden_units))

        out, _ = self.rnn(x, h)
        out = self.fc(out[:, -1, :])
        return out
        # for t in range(time_steps):
        #     xt = x[:, t, :]
        #     h = self.tanh(torch.matmul(xt, self.Wxh) + torch.matmul(h, self.Whh) + self.bh)
        #
        # y = torch.matmul(h, self.Why) + self.by
        # return y


X, y = create_dataset(num_sequences, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

input_dim = 1
hidden_units = 50
output_dim = 1


rnn = RNN(input_dim, hidden_units, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)


epochs = 10
for epoch in range(epochs):
    rnn.train()
    optimizer.zero_grad()

    output = rnn(X)
    loss = criterion(output, y.unsqueeze(1))

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


rnn.eval()
test_seq = torch.rand(1, sequence_length - 1, input_dim)
prediction = rnn(test_seq)

print("Input sequence:", test_seq.flatten().tolist())
print("Predicted next value:", prediction.item())
