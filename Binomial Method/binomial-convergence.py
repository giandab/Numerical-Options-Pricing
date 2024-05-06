from scipy.stats import norm
import numpy as np
from binomial_trees import binomial_tree_european
import matplotlib.pyplot as plt

def black_scholes_call(S,E,r,sigma,t,T):
    d1 = (np.log(S/E)+ (r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = (np.log(S/E)+ (r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*norm.cdf(d1) - E*(np.exp(-r*(T-t)))*norm.cdf(d2)

sigma = 0.2
S_0 = 105
p = 1/2
Strike = 110
r = 0.1



comparison = black_scholes_call(S_0,Strike,r,sigma,0,1)
binomial = []
M = [ x for x in range(2,600,5)]
print(M)

for N in M:
    delta_t = 1/N
    u = np.exp(r*delta_t)*(1 + np.sqrt(np.exp((sigma**2)*delta_t)-1))
    d = np.exp(r*delta_t)*(1 - np.sqrt(np.exp((sigma**2)*delta_t)-1))
    binomial.append(binomial_tree_european(Strike,S_0,u,d,p,r,delta_t,N))

print(binomial)
print(comparison)

plt.title("Comparison of Binomial and Analytical solution")
plt.plot(M,binomial,label= "Binomial Method")
plt.axhline(y=comparison, color='r', linestyle='-',label="Black-Scholes Solution")
plt.xlabel("Number of time-steps")
plt.ylabel("Option Price")
plt.legend()
plt.show()