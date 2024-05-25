import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize parameters
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def forward(self, inputs):
        # Initialize hidden state
        h_prev = np.zeros((self.hidden_size, 1))
        # Store intermediate values for backpropagation
        self.intermediates = {'h': [], 'x': [], 'y': []}
        
        for x in inputs:
            # Compute hidden state
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
            # Compute output
            y = np.dot(self.Why, h) + self.by
            
            # Store intermediate values
            self.intermediates['h'].append(h)
            self.intermediates['x'].append(x)
            self.intermediates['y'].append(y)
            
            # Update hidden state for next time step
            h_prev = h
            
        return np.array(self.intermediates['y'])
    
    def compute_loss(self, targets):
        # Compute loss
        loss = 0
        for y_pred, target in zip(self.intermediates['y'], targets):
            loss += np.sum((y_pred - target) ** 2)
        return loss / len(targets)
    
    def backward(self, targets, learning_rate):
        # Initialize gradients
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(self.intermediates['h'][0])
        
        # Backpropagate through time
        for t in reversed(range(len(targets))):
            dy = self.intermediates['y'][t] - targets[t]
            dWhy += np.dot(dy, self.intermediates['h'][t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - self.intermediates['h'][t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, self.intermediates['x'][t].T)
            dWhh += np.dot(dh_raw, self.intermediates['h'][t - 1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        # Clip gradients to prevent exploding gradients
        for gradient in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(gradient, -5, 5, out=gradient)
        
        # Update parameters
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby


if __name__ == "__main__":

    # 创建一个简单的时间序列数据集
    # 假设我们有一个长度为5的序列，每个时间步有一个长度为10的向量作为输入
    data = np.random.randn(5, 10)

    # 初始化RNN模型
    rnn = SimpleRNN(input_size=10, hidden_size=16, output_size=10)

    # 定义训练参数
    learning_rate = 0.01
    epochs = 1000

    # 训练模型
    for epoch in range(epochs):
        # 前向传播
        predictions = rnn.forward(data)
        
        # 计算损失
        loss = rnn.compute_loss(predictions)  # 使用模型的预测值来计算损失
        
        # 反向传播并更新参数
        rnn.backward(data, learning_rate)  # 损失计算和反向传播使用输入数据本身
        
        # 打印损失
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # 使用训练好的模型进行预测
    new_data = np.random.randn(5, 10)
    predictions = rnn.forward(new_data)
    print("Predictions:", predictions)
