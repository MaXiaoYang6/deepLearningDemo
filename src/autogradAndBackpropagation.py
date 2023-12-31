import torch
import torch.nn as nn

# 1) Design model ( input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute predictionf
#       - backward pass: gradients
#       - update weights


# f = w * x

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

# model prediction
input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

# gradient
# MSE = 1/N * (w*x - y) ** 2
# dJ/dw = 1/N * 2x (w*x - y)
# def gradient(x, y, y_predicted):
#     return np.dot(2 * x, y_predicted - y).mean()


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 30000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl / dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}, X = {X}, Y_pred = {y_pred}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
