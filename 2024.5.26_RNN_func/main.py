# encoding:utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# 设置随机种子
torch.manual_seed(3416)

# 设置超参数
TIME_STEP = 10
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LR = 0.02

class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, time_step, lr, initial_h_state=None):
        super(Rnn, self).__init__()
        self.time_step = time_step  # 添加time_step属性
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, 1)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.h_state = initial_h_state

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

        return steps, y_np, prediction

    def plot(self, steps, y_np, prediction):
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.show()

# 输入一个初始的h_state，返回最终的h_state
def func(initial_h_state):
    
    model = Rnn(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, TIME_STEP, LR, initial_h_state)
    steps, y_np, prediction = model.train(300)
    # model.plot(steps, y_np, prediction)
    return model.h_state


if __name__ == '__main__':

    initial_h_state = torch.tensor([[[ 0.2007,  0.0051,  0.4154,  0.4034,  0.3125, -0.0950, -0.8950,
            -0.7195, -0.0895,  0.3049,  0.9229, -0.3494,  0.2028, -0.7477,
            -0.2891,  0.0730, -0.9841, -0.1823, -0.8008,  0.3737, -0.4704,
            0.6192, -0.0636,  0.0086,  0.8629,  0.2117,  0.4237, -0.2888,
            0.0315,  0.6691, -0.7257, -0.7713]]])
    print(f"initial_h_state: {initial_h_state}")

    final_h_state = func(initial_h_state)
    print(f"final_h_state: {final_h_state}")