import numpy as np

LR = 0.01
def griden_SGD(theta, LR, X, Y, step):  # X is mxn  ; Y is mx1; theta is nx1
	h_x = X.dot(theta)  # mx1
	m = Y.size
	n = X.shape[1]
	for _ in range(step):
		for i in range(n):
			theta = theta + LR * np.subtract(Y[i], h_x[i]) * X[i].T
			return theta

x_data = [1,2,3,4]
y_data = [1.1,2.2,3.1,]

xs = np.array(x_data).reshape(-1, 1)
ys = np.array(y_data).reshape(-1, 1)

theta = griden_SGD(0, LR, xs, ys, 100)
print(theta)




