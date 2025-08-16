import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
from fredapi import Fred

from curl_cffi import requests
session = requests.Session(impersonate="chrome")

# main functions and calculations


# global variable to store the portfolio
current_portfolio = []

# getting the current portfolio input
def portfolio(inputs):
    global current_portfolio
    inputs = inputs.split(',')
    current_portfolio = [(inputs[i], float(inputs[i + 1])) for i in range(0, len(inputs), 2)]
    return current_portfolio

# getting the risk free rate from 10 year us gov bonds
def risk_free_rate():
    fred = Fred(api_key='your api key here')
    us_10yr_treasury = fred.get_series('DGS10')
    latest_rate = us_10yr_treasury.iloc[-1]
    risk_free_rate = latest_rate/100
    return risk_free_rate

# get all the daily returns on each stock
def all_returns(current_portfolio,t_trailing_back=None):

    # check if time frame has been set
    if t_trailing_back is not None:
        hany_ev = t_trailing_back
    else:
        hany_ev = 1

    minden_stock = [] 

    for ticker, share in current_portfolio:
        stock = yf.Ticker(ticker.strip(), session=session)
        
        # getting the data from timeframe
        one_year_ago = (datetime.now() - timedelta(days=float(hany_ev) * 365)).strftime("%Y-%m-%d")
        ttm_data = stock.history(start=one_year_ago)
        
        # adjusted closing price for dividend payments
        ttm_data['Adj Close'] = ttm_data['Close']
        for i in range(1, len(ttm_data)):
            dividend = ttm_data['Dividends'].iloc[i]
            if dividend > 0:
                ttm_data.loc[ttm_data.index[i], 'Adj Close'] += dividend
        
        # making sure no zero or negative values before calculating log returns
        ttm_data['log returns'] = np.log(ttm_data['Adj Close'] / ttm_data['Adj Close'].shift(1))
        ttm_data = ttm_data[ttm_data['Adj Close'] > 0]  # removing rows where adj close is zero or negative
        ttm_data.dropna(inplace=True)
        
        # add returns to the list
        minden_stock.append(ttm_data['log returns'])
    
    all_returns = minden_stock
    return all_returns

# compute daily portfolio returns
def daily_portfolio_returns(all_returns):
    daily_p_returns = all_returns.mean(axis=1)
    return daily_p_returns

# convert list of returns to DataFrame
def all_returns_df(current_portfolio,all_returns):
    returns_df = pd.DataFrame(all_returns).T
    returns_df.columns = [ticker for ticker, _ in current_portfolio]

    # interpolate over missing values
    returns_df = returns_df.interpolate(method='linear')
    return returns_df

# calculate current total portfolio market price
def calculate_portfolio_value(current_portfolio):
    total_portfolio_value = 0

    # get the latest share prices and multiply with the number of input shares
    for ticker, share in current_portfolio:
        stock = yf.Ticker(ticker.strip())
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        ttm_data = stock.history(start=one_year_ago)
        if ttm_data.empty:
            continue
        latest_price = ttm_data['Close'].iloc[-1]
        total_portfolio_value += latest_price * share

    return total_portfolio_value

# calculate expected return on current portfolio
def expected_return_on_portfolio(current_portfolio,all_returns_df,total_portfolio_value,t_project_returns=None):

    # check is timeframe been set
    if t_project_returns is None:
        t_project_returns = 1

    # list of all the stocks 
    all_returns_list = []
    weighted_returns = []

    for (i, (ticker, share)) in enumerate(current_portfolio):
        avg_return = all_returns_df[ticker].mean()
        daily_ER = np.exp(avg_return) - 1
        yearly_ER = ((1 + daily_ER) ** (float(t_project_returns) * 252)) - 1
    
        latest_price = all_returns_df.iloc[-1]
        stock_value = latest_price * share
        
        
        all_returns_list.append(yearly_ER)
        weighted_returns.append(yearly_ER * stock_value)
    
    portfolio_return = np.sum(weighted_returns) / total_portfolio_value
    return portfolio_return

# get portfolios covariance matrix
def get_cov_matrix(returns_df):
    cov_matrix = returns_df.cov()
    return cov_matrix

# get portfolio risk
def get_portfolio_std(current_portfolio,cov_matrix):
    weights = np.array([share for _, share in current_portfolio])
    weights = weights / sum(weights)
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix*252, weights))
    portfolio_std = np.sqrt(portfolio_var)
    return portfolio_std

# getting the sharpe ratio
def shapre_ratio(portfolio_return,portfolio_std,risk_free_rate):
    shapre = (portfolio_return - risk_free_rate) / portfolio_std
    return shapre


# extra functions and calculations


# get the highest and lowest daily volatility
def highest_lowest(daily_p_returns):
    min_return = min(daily_p_returns)
    max_return = max(daily_p_returns)
    return max_return, min_return

# go over 20000 random combinations of weights
def markovitz(risk_free_rate,returns_df):
    num_random_portf = 20000

    # calculate the stock returns for each
    avg_returns_on_each = returns_df.mean()
    yearly_log_returns = avg_returns_on_each * 252
    random_log_cov_matrix = returns_df.cov()


    # create the dataframe to store the randomly created data
    columns_for_random_portf = ['return', 'std', 'sharpe'] + ["Weight_" + str(ticker[0]) for ticker in returns_df.columns]
    random_portf_df = pd.DataFrame(columns=columns_for_random_portf)


    # create the random data
    for _ in range(num_random_portf):
        # assign the random weights, returns, std, and sharpe
        weights_random = np.random.random(len(returns_df.columns))
        weights_random /= np.sum(weights_random)

        random_expected_portfolio_return = np.dot(weights_random, yearly_log_returns)

        random_portfolio_risk = np.sqrt(np.dot(weights_random.T, np.dot(random_log_cov_matrix*252, weights_random)))

        random_sharpe_ratio = (random_expected_portfolio_return - risk_free_rate) / random_portfolio_risk
        
        # add them all to the df
        row_data = np.concatenate(([random_expected_portfolio_return, random_portfolio_risk, random_sharpe_ratio], weights_random))
        random_portf_df.loc[len(random_portf_df)] = row_data

    return random_portf_df

# function to get the highest sharpe rate on markov
def max_sharpe_portfolio(random_portf_df):
    max_sharpe_index = random_portf_df['sharpe'].idxmax()
    max_sharpe_portfolio = random_portf_df.loc[max_sharpe_index]
    return max_sharpe_portfolio

# function to get the lowest risk on markov
def min_risk_porfolio(random_portf_df):
    min_risk_index = random_portf_df['std'].idxmin()
    min_risk_portfolio = random_portf_df.loc[min_risk_index]
    return min_risk_portfolio

# gets the highest shapre allocation of your budget into the given portfolio
def max_sharpe_alloc(random_portf_df,initial_input_capital):
    # calculate the ideal money distribution
    max_sharpe_index = random_portf_df['sharpe'].idxmax()
    max_sharpe_portfolio = random_portf_df.loc[max_sharpe_index]
    w = max_sharpe_portfolio[3:]
    money_alloc = {ticker: f"{(weight * initial_input_capital):.2f} USD or {(weight * 100):.2f}%" for ticker, weight in w.items()}
    max_sharpe_money_alloc = pd.DataFrame(money_alloc.items(), columns=['ticker', 'allocated dough for max shapre'])

    return max_sharpe_money_alloc

# gets the lowest risk allocation of your budget into the given portfolio
def min_risk_alloc(random_portf_df,initial_input_capital):
    # calculate the ideal money distribution
    min_risk_index = random_portf_df['sharpe'].idxmin()
    min_risk_portfolio = random_portf_df.loc[min_risk_index]
    w = min_risk_portfolio[3:]
    money_alloc = {ticker: f"{(weight * initial_input_capital):.2f} USD or {(weight * 100):.2f}%" for ticker, weight in w.items()}
    min_risk_money_alloc = pd.DataFrame(money_alloc.items(), columns=['ticker', 'allocated dough for min risk'])

    return min_risk_money_alloc


# functions for excel analysis and debugging


# print cov_direction matrix into an excel file
def cov_direction_to_excel(cov_matrix):
     # covariance direction matrix to excel
    cov_direction_matrix = cov_matrix.map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cov_direction_matrix.index = [ticker for ticker, _ in current_portfolio]
    cov_direction_matrix.columns = [ticker for ticker, _ in current_portfolio]
    cov_direction_matrix.to_excel("your file destiantion and name here")

# print correlation matrix into an excel file
def corr_to_excel(cov_matrix):
    corr_matrix = cov_matrix.corr()
    corr_matrix.index = [ticker for ticker, _ in current_portfolio]
    corr_matrix.columns = [ticker for ticker, _ in current_portfolio]
    corr_matrix.to_excel("your file destiantion and name here")

# print the covariance matrix into an excel file
def cov_to_excel(cov_matrix):
    cov_matrix.to_excel("your file destiantion and name here")

# print all the data relating to each stock into an excel file
def ttm_data_to_excel(current_portfolio,t_trailing_back=None):

    # check if time frame has been set
    if t_trailing_back is None:
        t_trailing_back = 1

    minden_stock = [] 

    with pd.ExcelWriter("your file destiantion and name here") as writer:
        for ticker, share in current_portfolio:
            stock = yf.Ticker(ticker.strip())
            
            # getting the data from timeframe
            one_year_ago = (datetime.now() - timedelta(days=float(t_trailing_back) * 365)).strftime("%Y-%m-%d")
            ttm_data = stock.history(start=one_year_ago)
            
            # adjusted closing price for dividend payments
            ttm_data['Adj Close'] = ttm_data['Close']
            for i in range(1, len(ttm_data)):
                dividend = ttm_data['Dividends'].iloc[i]
                if dividend > 0:
                    ttm_data.loc[ttm_data.index[i], 'Adj Close'] += dividend
            
            # making sure no zero or negative values before calculating log returns
            ttm_data['log returns'] = np.log(ttm_data['Adj Close'] / ttm_data['Adj Close'].shift(1))
            ttm_data = ttm_data[ttm_data['Adj Close'] > 0]  # removing rows where adj close is zero or negative
            ttm_data.dropna(inplace=True)
            
            # add returns to the list
            minden_stock.append(ttm_data['log returns'])

            # write it into the data excel file
            no_time_zone_ttm = ttm_data.reset_index()
            no_time_zone_ttm['Date'] = no_time_zone_ttm['Date'].dt.tz_localize(None)
            no_time_zone_ttm.to_excel(writer, sheet_name=ticker)       

# prints the daily returns of the portfolio into an excel file
def returns_to_excel(daily_p_returns):
    # writing out daily portfolio returns into excel
    daily_portfolio_returns_to_excel = daily_p_returns.index.tz_localize(None)
    daily_portfolio_returns_to_excel = pd.DataFrame(daily_p_returns)
    daily_portfolio_returns_to_excel.to_excel("your file destiantion and name here", index=True)


# functions to visualize calculations


# plot the daily returns of the portfolio
def daily_returns_plt(daily_p_returns):
    # show the daily returns of the portfolio using matplotlib
    xaxis = range(len(daily_p_returns))
    plt.plot(xaxis, daily_p_returns)
    plt.ylabel("Daily % change")
    plt.xlabel(f"The last {len(daily_p_returns)} days")
    plt.show()

# function to plot the markovitz functions returns
def marko_scplt(random_portf_df):
    # scatterplot for the markovitz thing
    plt.scatter(random_portf_df['std']*100,random_portf_df['return']*100, c=random_portf_df['sharpe'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel("Portf std %")
    plt.ylabel("Portf return %")
    plt.show()
