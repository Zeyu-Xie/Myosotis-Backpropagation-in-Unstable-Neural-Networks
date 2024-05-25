import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.Wuu = np.random.randn(hidden_size, input_size) * 0.01
        self.Wxx = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wxy = np.random.randn(output_size, hidden_size) * 0.01

        # 初始化偏置
        self.bx = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, u, x_prev):
        """
        前向传播
        """
        self.u = u
        self.x_prev = x_prev

        # 计算隐藏状态
        self.x_next = np.tanh(np.dot(self.Wuu, u) +
                              np.dot(self.Wxx, x_prev) + self.bx)

        # 计算输出
        self.output = np.dot(self.Wxy, self.x_next) + self.by

        return self.output, self.x_next

    def backward(self, dy, learning_rate=0.001):
        """
        反向传播
        """
        # 计算输出层权重和偏置的梯度
        dWxy = np.dot(dy, self.x_next.T)
        dby = dy

        # 计算输出层输入的梯度
        dx_next = np.dot(self.Wxy.T, dy)

        # 计算隐藏层的梯度
        dx_raw = (1 - self.x_next ** 2) * dx_next

        # 计算隐藏层权重和偏置的梯度
        dWuu = np.dot(dx_raw, self.u.T)
        dWxx = np.dot(dx_raw, self.x_prev.T)
        dbx = dx_raw

        # 更新权重和偏置
        self.Wuu -= learning_rate * dWuu
        self.Wxx -= learning_rate * dWxx
        self.Wxy -= learning_rate * dWxy
        self.by -= learning_rate * dby
        self.bx -= learning_rate * dbx

    def train(self, inputs, targets, x_prev, learning_rate=0.001):
        """
        训练模型
        """
        loss = 0
        # 遍历输入序列
        for i in range(len(inputs)):
            # 前向传播
            output, x_prev = self.forward(inputs[i], x_prev)

            # 计算损失
            loss += np.mean((output - targets[i]) ** 2)

            # 反向传播
            dy = output - targets[i]
            self.backward(dy, learning_rate)

        return loss / len(inputs)


if __name__ == "__main__":

    inputs = [np.array([[0.1]]), np.array([[0.2]]), np.array([[0.3]])]
    targets = np.array([[0.2]])

    # 初始化 RNN 模型
    rnn = SimpleRNN(input_size=1, hidden_size=3, output_size=1)

    # 设置初始隐藏状态
    x_prev = np.zeros((3, 1))

    # 训练模型
    for epoch in range(1000):
        loss = rnn.train(inputs, targets, x_prev)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # 测试模型
    test_input = np.array([[0.4], [0.5], [0.6]])
    x_prev = np.zeros((3, 1))
    output, _ = rnn.forward(test_input, x_prev)
    print("Predicted Output:", output)
