import matplotlib.pyplot as plt

with open('wout') as f:
    cost_wout = [float(line.strip()) for line in f.readlines()]

with open('win') as f:
    cost_win = [float(line.strip()) for line in f.readlines()]

plt.plot(cost_wout, 'r')
plt.plot(cost_win, 'b')
plt.show()