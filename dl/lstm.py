import torch
import torch.nn as nn
import torch.optim as optim


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        self.Wx = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.01)
        self.Wh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, h, c):
        gates = torch.matmul(x, self.Wx) + torch.matmul(h, self.Wh) + self.b
        i, f, o, g = torch.split(gates, self.hidden_size, dim=1)

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)  # Candidate cell state

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], h, c)

        output = self.fc(h)
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 50
output_size = 1
time_steps = 10
batch_size = 32
epochs = 20

# model = LSTM(input_size, hidden_size, output_size)
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(1000, time_steps, input_size)
y = torch.randn(1000, output_size)

for epoch in range(epochs):
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i: i + batch_size]
        y_batch = y[i: i + batch_size]

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


test_sample = torch.randn(5, time_steps, input_size)
predictions = model(test_sample)
print("Predictions:", predictions)
