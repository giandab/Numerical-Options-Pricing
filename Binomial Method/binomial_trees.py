import numpy as np
from time import process_time

def payoff_call(strike,spot):
    return(max(spot-strike,0))

def binomial_tree_european(Strike,S_0,u,d,p,r,delta_t,N):

    discount = np.exp(-r*delta_t)

    asset_prices = [0]*(N+1)
    asset_prices[0] = S_0

    #Building Tree of asset prices
    for i in range(1,N+1):
        for x in range(i,0,-1):
            asset_prices[x] = u*asset_prices[x-1]
        asset_prices[0] = d*asset_prices[0]

    #Valuing option at each final price
    option_prices=[]
    for i in range(0,N+1):
        option_prices.append(payoff_call(Strike,asset_prices[i]))

    #Getting Present Values of option prices by walking backwards through tree and discounting
    for i in range(N,0,-1):
        for x in range(0,i):
            value = p*option_prices[x+1] + (1-p)*option_prices[x]
            option_prices[x]= discount*value


    return option_prices[0]

#For a call option expiring in 1 year.
sigma = 0.2
S_0 = 105

N = 100
p = 1/2
Strike = 110
r = 0.1
delta_t = 1/N

u = np.exp(r*delta_t)*(1 + np.sqrt(np.exp((sigma**2)*delta_t)-1))
d = np.exp(r*delta_t)*(1 - np.sqrt(np.exp((sigma**2)*delta_t)-1))

print(binomial_tree_european(Strike,S_0,u,d,p,r,delta_t,N))
