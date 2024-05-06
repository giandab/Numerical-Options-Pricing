import numpy as np
from time import process_time



def binomial_tree_european_fast(Strike,S_0,u,d,p,r,delta_t,N):

    discount = np.exp(-r*delta_t)

    #Directly obtaining final asset prices
    asset_prices = S_0*u**(np.arange(N,-1,-1))*d**(np.arange(0,N+1))

    #Valuing option at each final price
    option_prices = np.maximum( asset_prices-Strike , np.zeros(N+1))

    #Getting Present Values of option prices by walking backwards through tree and discounting
    for i in range(N,0,-1):
        option_prices = discount*(p*(option_prices[1:i+1]) + (1-p)*(option_prices[0:i]))


    return option_prices[0]

sigma = 0.2
S_0 = 105

N = 300
p = 1/2
Strike = 110
r = 0.1
delta_t = 1/N

u = np.exp(r*delta_t)*(1 + np.sqrt(np.exp((sigma**2)*delta_t)-1))
d = np.exp(r*delta_t)*(1 - np.sqrt(np.exp((sigma**2)*delta_t)-1))

print(binomial_tree_european_fast(Strike,S_0,u,d,p,r,delta_t,N))
