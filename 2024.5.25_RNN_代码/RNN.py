# encoding:utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class Rnn(nn.Module):
    def __init__(self, input_size, time_step, lr):
        super(Rnn, self).__init__()
        self.time_step = time_step
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.h_state = None

    def forward(self, x, h_0):
        r_out, h_n = self.rnn(x, h_0)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_n

    def train(self, steps):
        for step in range(steps):
            start, end = step * np.pi, (step + 1) * np.pi
            steps = np.linspace(start, end, self.time_step, dtype=np.float32)
            x_np = np.sin(steps)
            y_np = np.cos(steps)

            x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
            y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

            prediction, self.h_state = self(x, self.h_state)
            self.h_state = self.h_state.data

            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Step {step+1}/{steps}:")
            print("Input (x):", x)
            print("Target (y):", y)
            print("Prediction:", prediction)
            print("Loss:", loss.item())
            print("-----------------------")

        return steps, y_np, prediction

    def plot(self, steps, y_np, prediction):
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.show()


if __name__ == '__main__':
    TIME_STEP = 10
    INPUT_SIZE = 1
    LR = 0.02
    model = Rnn(INPUT_SIZE, TIME_STEP, LR)
    steps, y_np, prediction = model.train(300)
    model.plot(steps, y_np, prediction)
