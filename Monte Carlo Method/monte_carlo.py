import numpy as np
import matplotlib.pyplot as plt
from time import process_time
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

simulations = [x for x in range(100,1500,20)]
estimates = []

y = black_scholes_call(105,110,0.05,0.2,0,1)

for sim in simulations:
    estimates.append(monte_carlo(30,sim,0.05,105,0.2,110))

plt.title("Monte Carlo Option Price Estimates")
plt.plot(simulations,estimates,".")
plt.axhline(y, color = "r", linestyle="-",label="Black-Scholes solution")
plt.ylabel("Option Price")
plt.xlabel("Number of Monte Carlo Simulations")
plt.legend()
plt.show()

