import numpy as np
import pandas as pd
import datetime as dt
from io import StringIO

import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

class Stock:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def __str__(self):
        return f'Stock object for {self.ticker}'
    
    def __repr__(self):
        return f'Stock({self.ticker})'
    
    def add_data(self, data: pd.Series):
        self.data = data

class Basket:
    def __init__(self, tickerList: list, riskFreeRate: float):
        self.tickerList = sorted(tickerList)
        self.riskFreeRate = riskFreeRate
        self.stocks = list()
        for ticker in self.tickerList:
            stock = Stock(ticker)
            self.stocks.append(stock)
            
    def __str__(self):
        return f'Basket containing tickers: {self.stocks}'

    def __repr__(self):
        return f'Basket({self.stocks})'
    
    def __len__(self):
        return len(self.stocks)
    
    def get_data(self, start: dt.date, end: dt.date):
        self.start = start
        self.end = end
        try:
            data = yf.download(self.tickerList, start=start, end=end, progress=False)['Adj Close']
            self.data = data
            # Assign data to Stock objects
            if len(self.tickerList) == 1:
                self.stocks[0].add_data(data)
            elif len(self.tickerList) > 1:
                for stock, ticker in zip(self.stocks, self.tickerList):
                    stock.add_data(data[ticker])
        except Exception as e:
            print(f'Error: {e}')

    def calculate_mv(self, investment_start: dt.date):
        '''
        Calculate mean returns and covariance matrix for the basket of stocks over the period ending at the investment_start date.
        '''
        returns = self.data[:investment_start].pct_change().dropna()
        mean_returns = returns.mean()*252
        self.mean_returns = mean_returns
        if len(self) == 1:
            cov_matrix = returns.var()*252
            self.cov_matrix = cov_matrix
        else:
            cov_matrix = returns.cov()*252
            self.cov_matrix = cov_matrix

    def get_mv(self):
        return self.mean_returns, self.cov_matrix

    def mv_analysis(self, investment_start: dt.date):
        '''
        Run the mean-variance analysis for the basket of stocks over the period ending at the investment_start date.

        Returns:
        stocks_mv: Dataframe
        minVar_portfolio: Dictionary
        max_sharpe_portfolio: Dictionary
        efficient_frontier: Dataframe
        '''
        returns = self.data[:investment_start].pct_change().dropna()
        mean_returns = returns.mean()*252
        self.mean_returns = mean_returns
        if len(self) == 1:
            cov_matrix = returns.var()*252
            self.cov_matrix = cov_matrix
            stocks_mv = pd.DataFrame({'Return': mean_returns, 'Risk': np.sqrt(cov_matrix)}, index=self.tickerList)
            minVar_portfolio = None
            maxSharpe_portfolio = None
            efficient_frontier = None
        else:
            cov_matrix = returns.cov()*252
            self.cov_matrix = cov_matrix
            stocks_mv = pd.DataFrame({'Return': mean_returns, 'Risk': np.sqrt(np.diag(cov_matrix))}, index=self.tickerList)
            minVar_portfolio = self.minVar_portfolio()
            maxSharpe_portfolio = self.maxSharpe_portfolio()
            # Find the max value of mu for the efficient frontier
            y_max = max(maxSharpe_portfolio['return'], stocks_mv['Return'].max())
            efficient_frontier = self.minimum_variance_line(np.linspace(0,y_max*1.1, 500))
        
        return stocks_mv, minVar_portfolio, maxSharpe_portfolio, efficient_frontier
    
    def minVar_portfolio(self):
        nAssets = len(self)
        m = self.mean_returns
        C = self.cov_matrix
        u = np.ones(nAssets)

        # Intermediate calculations
        C_inv = np.linalg.inv(C)
        w = C_inv.dot(u) / u.T.dot(C_inv).dot(u)
        mu = m.T.dot(w)
        sigma = np.sqrt(w.T.dot(C).dot(w))

        minVar_portfolio = {'weights': w, 'return': mu, 'risk': sigma}

        return minVar_portfolio

    def maxSharpe_portfolio(self):
        '''
        Calculate the max Sharpe portfolio for a given set of mean returns and covariance matrix.
        '''
        nAssets = len(self)
        m = self.mean_returns
        C = self.cov_matrix
        u = np.ones(nAssets)
        r = self.riskFreeRate

        # Intermediate calculations
        C_inv = np.linalg.inv(C)
        w = C_inv.dot(m - r*u) / (m - r*u).T.dot(C_inv).dot(u)
        mu = m.T.dot(w)
        sigma = np.sqrt(w.T.dot(C).dot(w))

        max_sharpe_portfolio = {'weights': w, 'return': mu, 'risk': sigma}

        return max_sharpe_portfolio

    def minimum_variance_line(self, mu):
    
        nAssets = len(self)
        m = self.mean_returns
        C = self.cov_matrix
        u = np.ones(nAssets)

        # Intermediate calculations
        C_inv = np.linalg.inv(C)
        D_mat = np.array([[u.T.dot(C_inv).dot(u), u.T.dot(C_inv).dot(m)],
                        [m.T.dot(C_inv).dot(u), m.T.dot(C_inv).dot(m)]])
        D = np.linalg.det(D_mat)
        a = (u.T.dot(C_inv).dot(u)*C_inv.dot(m) - u.T.dot(C_inv).dot(m)*C_inv.dot(u)) / D
        b = (m.T.dot(C_inv).dot(m)*C_inv.dot(u) - m.T.dot(C_inv).dot(u)*C_inv.dot(m)) / D

        # Calculate the minimum variance line
        w_mu = np.zeros((nAssets, len(mu)))
        sigma_mu = np.zeros_like(mu)
        for i, param in enumerate(mu):
            w_mu[:,i] = a*param + b
            sigma_mu[i] = np.sqrt(w_mu[:,i].T.dot(C).dot(w_mu[:,i]))

        minVar_line = pd.DataFrame({'Return': mu, 'Risk': sigma_mu, **{f'{ticker} weight': w_mu[i] for i, ticker in enumerate(self.tickerList)}})

        return minVar_line
    
class Portfolio:
    def __init__(self, basket: Basket, riskFreeRate: float, includeRiskFree: bool):
        self.basket = basket
        self.riskFreeRate = riskFreeRate
        self.includeRiskFree = includeRiskFree

    def __repr__(self):
        substr = 'risk-free asset included' if self.includeRiskFree else ''
        return f'Portfolio({self.basket}, r = {self.riskFreeRate}, {substr})'
    
    def __len__(self):
        return len(self.basket) + 1 if self.includeRiskFree else len(self.basket)
    
    def set_investment_start(self, investment_start: dt.date):
        self.investment_start = investment_start
    
    def get_analysis(self):
        data = self.basket.data
        self.basket.calculate_mv(self.investment_start)
        mean_returns, cov_matrix = self.basket.get_mv()
        if self.includeRiskFree:
            mean_returns.loc['Risk-free'] = self.riskFreeRate
            cov_matrix.loc['Risk-free'] = 0
            cov_matrix['Risk-free'] = 0
        return mean_returns, cov_matrix
    

    def generate(self, numPortfolios: int, shortSelling: bool):
        mean_returns, cov_matrix = self.get_analysis()
        tickers = mean_returns.index
        nAssets = len(tickers)

        # Create an empty DataFrame to store the results
        mcPortfolios = pd.DataFrame(columns=[ticker+' weight' for ticker in tickers] + ['Return', 'Risk', 'Sharpe Ratio', 'Short positions'], index=range(numPortfolios), dtype=float)

        # Generate random weights and calculate the expected return, volatility and Sharpe ratio
        for i in range(numPortfolios):
            weights = 2*np.random.random(nAssets) - 1 if shortSelling else np.random.random(nAssets)
            weights /= np.sum(weights)
            mcPortfolios.loc[i, [ticker+' weight' for ticker in tickers]] = weights

            # Calculate the expected return
            mcPortfolios.loc[i, 'Return'] = np.dot(weights, mean_returns)

            # Calculate the expected volatility
            mcPortfolios.loc[i, 'Risk'] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate the Sharpe ratio
        mcPortfolios['Sharpe Ratio'] = (mcPortfolios['Return'] - self.riskFreeRate) / mcPortfolios['Risk']
        # Set flag for short positions by checking if any weight is negative
        mcPortfolios['Short positions'] = mcPortfolios[[ticker+' weight' for ticker in tickers]].apply(lambda x: any(x<0), axis=1)

        self.mcPortfolios = mcPortfolios

        return mcPortfolios
    
    # Load from json
    def load(self, data):
        '''
        Load the portfolio from a json file
        '''
        self.mcPortfolios = pd.read_json(StringIO(data))
    
    def evaluate(self, index: int, initialValue: float):
        '''
        Evaluate the portfolio
        '''
        sample_portfolio = self.mcPortfolios.iloc[index]
        data = self.basket.data[self.investment_start:]
        r = self.riskFreeRate
        assets = self.basket.tickerList
        nShares = sample_portfolio[[asset+' weight' for asset in assets]].rename({asset+' weight': asset for asset in assets})*initialValue/data.iloc[0]
        if self.includeRiskFree:
            cash = sample_portfolio['Risk-free weight']*initialValue*pd.Series((1+r)**(np.arange(len(data))/252), index=data.index)
            portfolio_value = nShares.dot(data.T) + cash
        else:
            portfolio_value = nShares.dot(data.T)
        return portfolio_value
