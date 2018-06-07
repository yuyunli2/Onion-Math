import numpy as np
import matplotlib.pyplot as plt

# Initialize
N, n_in, n_out = 640, 10, 1
learning_rate = 1e-4

x = np.random.randn(N,  n_in)
y = 3 * np.sum(x, axis = 1).reshape(N, n_out)
w1 = np.random.randn(n_in, n_out)
w2 = np.random.randn(n_in, n_out)

loss_list1 = []
loss_list2 = []

for i in range(5000):
    h = x.dot(w1)
    y_hat = h
    y_hat[h <0] = 0
    loss = np.sum(0.5 * (y_hat - y)**2)
    grad_y_pred = y_hat - y
    grad_y_pred[grad_y_pred < 0] = 0
    grad_y_pred[grad_y_pred > 0] = 1
    grad_w = x.T.dot(grad_y_pred)
    w1 -= learning_rate * grad_w
    loss_list1.append(loss)

for i in range(5000):
    h = x.dot(w2)
    y_hat = 1/(1 + np.exp(-h))
    loss = np.sum(0.5 * (y_hat - y)**2)
    grad_y_pred = y_hat - y
    grad_w = x.T.dot(grad_y_pred * (y_hat - y_hat**2))
    w2 -= learning_rate * grad_w
    loss_list2.append(loss)

# Plot
x = np.arange(1, 5001, 1)
plt.title('Loss funciton')
plt.xlabel('Times')
plt.ylabel('Loss')
p1, = plt.plot(x, loss_list1, label='Relu')
p2, = plt.plot(x, loss_list2, label='Sigmoid')
plt.legend([p1, p2], ['Relu', 'Sigmoid'])


###############################################################
# Add one hidden layer

# Initialize
N, n_in, n_hidden, n_out = 640, 10, 32, 1
learning_rate = 1e-4
x = np.random.randn(N,  n_in)
y = 3 * np.sum(x, axis = 1).reshape(N, n_out)
print('y.shape', y.shape)
w3 = np.random.randn(n_in, n_hidden)
w4 = np.random.randn(n_hidden, n_out)

loss_list = []
for i in range(5000):
    h1 = x.dot(w3)
#     print('h1', h1.shape)
    h1[h1 <0] = 0
    h2 = h1.dot(w4)
    y_hat = 1/(1 + np.exp(-h2))
    loss = np.sum(0.5 * (y_hat - y)**2)
    grad_y_pred = y_hat - y
    grad_g_pred = grad_y_pred * (y_hat - y_hat**2)
#     print('grad_g_pred', grad_g_pred.shape)
    grad_f_pred = grad_g_pred
    grad_f_pred[grad_f_pred < 0] = 0
    grad_f_pred[grad_f_pred > 0] = 1
    grad_w = x.T.dot(grad_g_pred * grad_f_pred)
    w3 -= learning_rate * grad_w
    loss_list.append(loss)

# print(loss_list[4999])
x = np.arange(1, 5001, 1)
plt.title('Relu + Sigmoid')
p1, = plt.plot(x, loss_list)
