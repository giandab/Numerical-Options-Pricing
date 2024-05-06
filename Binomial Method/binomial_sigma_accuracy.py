import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S,E,r,sigma,t,T):

    d1 = (np.log(S/E)+ (r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = (np.log(S/E)+ (r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    return S*norm.cdf(d1) - E*(np.exp(-r*(T-t)))*norm.cdf(d2)

def binomial_tree_european_fast(Strike,S_0,u,d,p,r,delta_t,N):

    discount = np.exp(-r*delta_t)

    #Directly obtaining final asset prices
    asset_prices = S_0*u**(np.arange(N,-1,-1))*d**(np.arange(0,N+1))

    #Valuing option at each final price
    option_prices = np.maximum( asset_prices - Strike , np.zeros(N+1))

    #Getting Present Values of option prices by walking backwards through tree and discounting
    for i in range(N,0,-1):
        option_prices = discount*(p*(option_prices[1:i+1]) + (1-p)*(option_prices[0:i]))


    return option_prices[0]


S_0 = 105

N = 300
p = 1/2
Strike = 110
r = 0.1
delta_t = 1/N

accuracy = []
sigma = [x/100 for x in range(50,800)]

for value in sigma:
    u = np.exp(r*delta_t)*(1 + np.sqrt(np.exp((value**2)*delta_t)-1))
    d = np.exp(r*delta_t)*(1 - np.sqrt(np.exp((value**2)*delta_t)-1))

    binomial_value= (binomial_tree_european_fast(Strike,S_0,u,d,p,r,delta_t,N))
    black_scholes_value = (black_scholes_call(S_0,Strike,r,value,0,1))

    accuracy.append(binomial_value - black_scholes_value)

plt.title("Accuracy of Binomial Model as sigma varies")
plt.plot(sigma, accuracy)
plt.xlabel("Volatility (sigma)")
plt.ylabel("Binomial - BSM")
plt.legend()
plt.show()