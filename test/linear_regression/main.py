import numpy as np

# y = wx + b
def compute_loss(b, w, points):
    totalLoss = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # calculate mean square error
        totalLoss += (y - (w * x + b)) ** 2
    # average loss for each point
    return totalLoss / float(len(points))


# 计算每一步的梯度并更新
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(2x+b-y)
        b_gradient += (2/N) * ((w_current*x + b_current) - y)
        w_gradient += (2/N) * ((w_current*x + b_current) - y) * x

    # update w' & b'
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


# 对w和b进行多轮更新
def epoch(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000

    print("The initial b= {}, w= {}, loss={}".format(
        initial_b, initial_w, compute_loss(initial_b, initial_w, points)
    ))
    print("running...")

    [b, w] = epoch(points, initial_b, initial_w, lr, num_iterations=num_iterations)

    print("The final b= {}, w= {}, loss={}".format(
        b, w, compute_loss(b, w, points)
    ))


if __name__ == '__main__':
    run()