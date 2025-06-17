import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ダミー株価データ（sin波で近似）
days = np.arange(0, 100, 1)
prices = np.sin(days * 0.1) + np.random.normal(scale=0.1, size=len(days))

# 入力(x)は過去5日、出力(y)は翌日価格
# スライディングウィンドウ法
X = []
y = []

window_size = 5
for i in range(len(prices) - window_size):
    X.append(prices[i:i+window_size])
    y.append(prices[i+window_size])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

print("------- Data Summary -------")
print("X: " + str(X))
print("y: " + str(y))

    
# NumPy → PyTorchテンソル
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("------- Tensor Summary -------")
print("X_tensor: " + str(X_tensor))
print("y_tensor: " + str(y_tensor))
# データの形状確認
print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)

# 学習：80%、検証：20%
train_size = int(len(X_tensor) * 0.8)
X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]
X_val = X_tensor[train_size:]
y_val = y_tensor[train_size:]

# MLPモデルの定義
class StockMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # 出力は1つ（価格）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# モデルの初期化、損失関数、オプティマイザの設定
model = StockMLP()
criterion = nn.MSELoss()  # 平均二乗誤差（回帰向け）
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習ループ
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# 学習済みモデルの保存
torch.save(model.state_dict(), "stock_mlp_model.pth")

#モデルを表示
# state_dict = torch.load('stock_mlp_model.pth')

# for name, param in state_dict.items():
#     print(f"{name}: shape={param.shape}")
#     print(param)  # 実際の数値も見たい場合
#     print()


# モデルの評価
model.eval()
predicted = model(X_tensor).detach().numpy()

plt.plot(prices[window_size:], label="True Price")
plt.plot(predicted, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction with MLP")
plt.show()
