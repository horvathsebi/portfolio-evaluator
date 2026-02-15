import pandas as pd
from scipy.stats import norm
import numpy as np
import time
from curl_cffi import requests

class PortfolioEvaluator:
    def __init__(self, tickers, weights=None, risk_free_rate=0.04):

        self.tickers = [t.strip().upper() for t in tickers]
        self.risk_free_rate = risk_free_rate
        
        if not weights:
            n = len(self.tickers)
            self.weights = np.array([1/n] * n)
        else:
            self.weights = np.array(weights)

            if np.sum(self.weights) != 0:
                self.weights /= np.sum(self.weights)
            
        self.returns_df = pd.DataFrame()
        self.cov_matrix = None
        self.hist_mean_return = 0.0
        self.std_dev = 0.0
        self.sharpe_ratio = 0.0
        self.var_95 = 0.0  
        self.cvar_95 = 0.0 

    def fetch_data(self, period="1y"):
        """
        Fetches data directly using Yahoo Finance API instead of yfinance lib.
        """
        if not self.tickers:
            return None
        
        all_log_returns = {}
        valid_tickers = []
        session = requests.Session(impersonate="chrome")

        for ticker in self.tickers:
            try:
                time.sleep(0.1)
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval=1d&events=div"
                response = session.get(url)
                
                if response.status_code != 200:
                    continue

                data = response.json()
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                dates = pd.to_datetime(timestamps, unit='s')
                closes = result['indicators']['quote'][0]['close']
                
                df = pd.DataFrame({'Close': closes}, index=dates)
                
                df['Adj Close'] = df['Close']
                
                if 'events' in result and 'dividends' in result['events']:
                    div_data = result['events']['dividends']
                    for ts, div_info in div_data.items():
                        div_date = pd.to_datetime(int(ts), unit='s')
                        if div_date in df.index:
                            df.loc[div_date, 'Adj Close'] += div_info['amount']
                
                df = df.dropna(subset=['Adj Close'])
                df = df[df['Adj Close'] > 0]

                df['log_ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
                df.dropna(subset=['log_ret'], inplace=True)
                
                if not df.empty:
                    all_log_returns[ticker] = df['log_ret']
                    valid_tickers.append(ticker)
                    
            except Exception as e:
                print(f"Error parsing {ticker}: {e}")

        session.close()

        if not all_log_returns:
            return None

        self.returns_df = pd.DataFrame(all_log_returns).interpolate(method='linear').dropna()
        self.tickers = valid_tickers
        
        if len(self.weights) != len(self.tickers):
             self.weights = np.array([1/len(self.tickers)] * len(self.tickers))
        
        self.cov_matrix = self.returns_df.cov() * 252
        

        self.std_dev = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        self.hist_mean_return = np.sum(self.returns_df.mean() * self.weights) * 252
        
        if self.std_dev != 0:
            self.sharpe_ratio = (self.hist_mean_return - self.risk_free_rate) / self.std_dev

        
        confidence_level = 0.05
        self.var_95 = norm.ppf(confidence_level, self.hist_mean_return, self.std_dev)
        
        return self.returns_df

    def get_correlation_matrix(self):
        return self.returns_df.corr() if not self.returns_df.empty else pd.DataFrame()

    def run_monte_carlo(self, num_simulations=5000):

        if self.returns_df.empty:
            return pd.DataFrame()

        mean_returns = self.returns_df.mean().values * 252
        cov_matrix = self.cov_matrix.values
        num_assets = len(self.tickers)

        weights = np.random.random((num_simulations, num_assets))

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

        port_returns = np.dot(weights, mean_returns)

        port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
        
        sharpe_ratios = (port_returns - self.risk_free_rate) / port_vols


        df = pd.DataFrame({
            'Return': port_returns,
            'Volatility': port_vols,
            'Sharpe': sharpe_ratios
        })

        weight_df = pd.DataFrame(weights, columns=self.tickers)
        df = pd.concat([df, weight_df], axis=1)

        return df