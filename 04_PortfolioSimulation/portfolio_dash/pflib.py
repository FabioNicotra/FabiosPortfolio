import numpy as np
import pandas as pd
import datetime as dt

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
            cov_matrix = returns.cov()*252 if len(self) > 1 else returns.var()*252
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