import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S,E,r,sigma,t,T):

    d1 = (np.log(S/E)+ (r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = (np.log(S/E)+ (r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    return S*norm.cdf(d1) - E*(np.exp(-r*(T-t)))*norm.cdf(d2)

def monte_carlo(N,simulations,r,S_0,sigma,Strike):

    delta_t = 1/N

    def payoff(asset_price,strike_price):
        return max(asset_price-strike_price,0)

    #Obtaining Price paths for the underlying asset
    def asset_prices(S_0,N,delta_t,sigma,r,simulations):

        asset_price_paths = []

        for i in range(simulations):
            
            Single_path = [S_0]

            for x in range(1,N):

                epsilon = np.random.normal()
                Single_path.append(Single_path[x-1]+Single_path[x-1]*(r*delta_t + sigma*epsilon*np.sqrt(delta_t)))

            asset_price_paths.append(Single_path)

        return asset_price_paths

    #Obtaining option prices at expiry, returning the mean
    def option_prices(asset_prices,Strike):

        option_prices = []

        for path in asset_prices:
            option_prices.append(payoff(path[-1],Strike))

        return np.mean(option_prices)

    #discounting the mean option value at expiry
    discount = np.exp(-r)

    asset_price_paths = asset_prices(S_0,N,delta_t,sigma,r,simulations)
    mean_option_price = option_prices(asset_price_paths,Strike)

    discounted_estimate = mean_option_price*discount

    return discounted_estimate

simulations = 1000
N = 30

sigma = [x/100 for x in range(1,400)]

black_scholes_values = np.zeros(len(sigma))
monte_carlo_values = np.zeros(len(sigma))


for i in range(len(sigma)):

    black_scholes_values[i] = black_scholes_call(105,110,0.1,sigma[i],0,1)
    monte_carlo_values[i] = monte_carlo(N,simulations,0.1,105,sigma[i],110)

difference = monte_carlo_values - black_scholes_values


plt.title("Convergence of Monte Carlo Method relative to volatility")
plt.xlabel("Volatility (sigma)")
plt.ylabel("Error")
plt.plot(sigma,difference)
plt.show()
