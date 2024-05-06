from fast_binomial import binomial_tree_european_fast
from binomial_trees import binomial_tree_european
from time import process_time
import matplotlib.pyplot as plt
import numpy as np


p = 1/2
Strike = 110
r = 0.1
sigma = 0.2
S_0 = 105

original_algorithm_times = []
new_algorithm_times = []

M = [x for x in range(100,1500,10)]

for N in M:
    delta_t = 1/N
    u = np.exp(r*delta_t)*(1 + np.sqrt(np.exp((sigma**2)*delta_t)-1))
    d = np.exp(r*delta_t)*(1 - np.sqrt(np.exp((sigma**2)*delta_t)-1))

    print(N)

    start = process_time()
    binomial_tree_european(Strike,S_0,u,d,p,r,delta_t,N)
    end = process_time()

    original_algorithm_times.append(end-start)

    start = process_time()
    binomial_tree_european_fast(Strike,S_0,u,d,p,r,delta_t,N)
    end = process_time()

    new_algorithm_times.append(end-start)

plt.title("Comparing Algorithm Process Time")
plt.plot(M,original_algorithm_times,label="Algorithm 3.1")
plt.plot(M,new_algorithm_times,label="Algorithm 3.2 (new)")
plt.ylabel("Execution time (s)")
plt.xlabel("Number of Time steps")
plt.legend()
plt.show()


