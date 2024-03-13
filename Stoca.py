# %% libs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %% generation
N = 100  # numbero of realizations
n = 500  # number of observations
t = np.linspace(0, 1, n)  # times
dt = 1/n  # the time division
z = np.sqrt(dt)*np.random.randn(n-1, N)  # white noise
W = np.zeros([1, N])  # brownian motion starting point
# the brownian motion approximation as a random walk
W = np.cumsum(np.concatenate((W, z), axis=0), axis=0)
eps = np.diff(W, axis=0)
s = range(50, n, 100)
a = W[s].T  # selecting a set of realizations/rows and trasposing
vart = np.var(a, axis=0)  # variance estimation as funcion of time
# %% viz
plt.subplot(2, 2, 1)
plt.plot(t, W, linewidth=0.7)
plt.title('Brownian motions')
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(2, 2, 3)
plt.plot(t[0:-1], eps[:, 0:2], linewidth=0.7)
plt.title('White Noise')
plt.subplot(2, 2, 2)
# kernel density estimation at various times, in order to see the increase in variance
for i in range(a.shape[1]):
    # $ $ sintassi per usare LateX
    sns.kdeplot(a[:, i], label=r'$\sigma^{2}_{t}= $' +
                np.array2string(np.round(vart[i], decimals=2)))
plt.legend(fontsize=6)
plt.subplot(2, 2, 4)
sns.kdeplot(eps[:, 0:5])  # kernel density estimation of white noise
plt.tight_layout()  # aggiusta automaticamento lo spazio tra i subplot
# %%
print(vart)
