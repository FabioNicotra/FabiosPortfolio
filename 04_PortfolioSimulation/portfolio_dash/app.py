import numpy as np
import pandas as pd
import datetime as dt

import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

from io import StringIO

from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate

def generate_portfolios(returns, numPortfolios, riskFreeRate=0, shortSelling=False):
    tickers = returns.columns
    nAssets = len(tickers)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252


    # Create an empty DataFrame to store the results
    portfolios = pd.DataFrame(columns=[ticker+' weight' for ticker in tickers] + ['Return', 'Risk', 'Sharpe Ratio'], index=range(numPortfolios), dtype=float)

    # Generate random weights and calculate the expected return, volatility and Sharpe ratio
    for i in range(numPortfolios):
        weights = np.random.random(nAssets)
        weights /= np.sum(weights)
        portfolios.loc[i, [ticker+' weight' for ticker in tickers]] = weights

        # Calculate the expected return
        portfolios.loc[i, 'Return'] = np.dot(weights, mean_returns)

        # Calculate the expected volatility
        portfolios.loc[i, 'Risk'] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate the Sharpe ratio
    portfolios['Sharpe Ratio'] = (portfolios['Return'] - riskFreeRate) / portfolios['Risk']

    return portfolios
    
def evaluate_portfolio(mc_portfolios, index, data, initialValue):
    portfolio = mc_portfolios.loc[index]
    tickers = data.columns
    nShares = portfolio[[ticker+' weight' for ticker in tickers]].rename({ticker+' weight' : ticker for ticker in tickers})*initialValue/data.iloc[0]
    portfolio_value = nShares.dot(data.T)
    return portfolio_value

def evaluate_asset(tickers, index, data, initialValue):
    asset = data.iloc[:, index] if len(tickers) > 1 else data
    nShares = initialValue/asset.iloc[0]
    asset_value = nShares*asset
    return asset_value

def minimum_variance_portfolio(mean_returns, cov_matrix):
    nAssets = len(mean_returns)
    m = mean_returns
    C = cov_matrix
    u = np.ones(nAssets)

    # Intermediate calculations
    C_inv = np.linalg.inv(C)
    w = C_inv.dot(u) / u.T.dot(C_inv).dot(u)
    mu = m.T.dot(w)
    sigma = np.sqrt(w.T.dot(C).dot(w))

    return mu, sigma

def minimum_variance_line(mean_returns, cov_matrix):
    
    nAssets = len(mean_returns) if mean_returns.ndim > 0 else 1
    m = mean_returns
    C = cov_matrix
    u = np.ones(nAssets)

    # Intermediate calculations
    C_inv = np.linalg.inv(C)
    D_mat = np.array([[u.T.dot(C_inv).dot(u), u.T.dot(C_inv).dot(m)],
                      [m.T.dot(C_inv).dot(u), m.T.dot(C_inv).dot(m)]])
    D = np.linalg.det(D_mat)
    a = (u.T.dot(C_inv).dot(u)*C_inv.dot(m) - u.T.dot(C_inv).dot(m)*C_inv.dot(u)) / D
    b = (m.T.dot(C_inv).dot(m)*C_inv.dot(u) - m.T.dot(C_inv).dot(u)*C_inv.dot(m)) / D

    # Calculate the minimum variance line
    mu = np.linspace(0, 0.4, 100)
    w_mu = np.zeros((nAssets, len(mu)))
    sigma_mu = np.zeros(len(mu))
    for i, param in enumerate(mu):
        w_mu[:,i] = a*param + b
        sigma_mu[i] = np.sqrt(w_mu[:,i].T.dot(C).dot(w_mu[:,i]))

    return mu, sigma_mu

def market_portfolio(mean_returns, cov_matrix, riskFreeRate=0):
    nAssets = len(mean_returns)
    m = mean_returns
    C = cov_matrix
    u = np.ones(nAssets)

    # Intermediate calculations
    C_inv = np.linalg.inv(C)
    w = C_inv.dot(m - riskFreeRate*u) / (m - riskFreeRate*u).T.dot(C_inv).dot(u)
    mu = m.T.dot(w)
    sigma = np.sqrt(w.T.dot(C).dot(w))

    return mu, sigma

# Get a list of symbols from FTSEMIB index
ftsemib = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')[1]
ftsemib['ICB Sector'] = ftsemib['ICB Sector'].str.extract(r'\((.*?)\)', expand=False).fillna(ftsemib['ICB Sector'])


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("CERULEAN")

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN, dbc_css])

app.layout = html.Div([
    dcc.Store(id='store-data'),
    dcc.Store(id='store-portfolios'),
    html.Div(className='dbc',
             children=[
                 dbc.Row(
                     children=[
                         dbc.Col(
                             width=2,
                             children=[dbc.Card([
                                 html.P('Select assets', className='dbc'),
                                 dcc.Dropdown(
                                     id='ticker-dropdown',
                                     options=[
                                         {'label': f"{row['Company']} ({row['Ticker']})", 'value': row['Ticker']}
                                         for _, row in ftsemib.iterrows()
                                     ],
                                     multi=True,
                                     className='dbc'
                                 ),
                                 html.Br(),

                                 html.P('Select start date',
                                        className='dbc'),
                                 dcc.DatePickerSingle(
                                     id='start-date',
                                     min_date_allowed=dt.date(2010, 1, 1),
                                     max_date_allowed=dt.date.today() - dt.timedelta(days=365),
                                     initial_visible_month=dt.date.today() - dt.timedelta(days=365),
                                     date=dt.date.today() - dt.timedelta(days=365),
                                     display_format='DD/MM/YYYY',
                                     className='dbc'
                                 ),

                                 html.Br(),
                                 html.P('Analysis window',
                                        className='dbc'),
                                 dcc.Slider(
                                     id='analysis-window',
                                     min=1,
                                     max=10,
                                     step=1,
                                     value=1,
                                     marks=None,
                                     tooltip={
                                         'placement': 'bottom',
                                         'always_visible': True,
                                         'template': '{value} years',
                                     },
                                     className='dbc'
                                 ),
                                 html.P('Number of samples', className='dbc'),
                                 dbc.Input(id='n-portfolios', value=1000, type='number', className='dbc'),

                                 dbc.Button('Generate',
                                            id='generate-button',
                                            n_clicks=0,
                                            className='dbc'),

                                 html.Br(),

                                 html.P('Initial investment', className='dbc'),
                                 dbc.Input(id='initial-investment', value=100, type='number', className='dbc'),
                             ])]
                         ),
                         dbc.Col(
                             width=10,
                             children=[
                                 dcc.Tabs(
                                     id='tabs',
                                     value='tab-1',
                                     className='dbc',
                                     children=[
                                         dcc.Tab(
                                             label='Portfolio selection',
                                             id='tab-1',
                                             className='dbc',
                                             children=[
                                                 dbc.Row([

                                                         dbc.Col(
                                                            children = dcc.Graph(id='markowitz-graph',
                                                                       clear_on_unhover=True)
                                                         ),

                                                         dbc.Col(
                                                             children=[
                                                                 dcc.Graph(id='portfolio-value',
                                                                   )
                                                             ]
                                                         )
                                                 ]),

                                             ]
                                         ),
                                         dcc.Tab(
                                             id='tab-3',
                                             label='Realized returns',
                                             children=[
                                                 html.Div('Content tab 3', className='dbc')
                                             ]
                                         ),
                                     ]
                                 ),
                             ]
                         ),
                     ],
                 )
             ]
             ),


])


# Download data and plot mean-variance graph for selected assets
@app.callback(
    [Output('markowitz-graph', 'figure'),
     Output('store-data', 'data')],
    [Input('ticker-dropdown', 'value'),
     Input('start-date', 'date'),
     Input('analysis-window', 'value')],
    # prevent_initial_call=True,
    # suppress_callback_exceptions=True
)
def select_assets(tickers, investment_start_date, window, riskFreeRate=0.05):
    if not tickers:
        fig = go.Figure().update_xaxes(title='Risk', range=[0, 0.5]).update_yaxes(title='Return',range=[0, 0.4]).update_layout(transition_duration=500)
        data = pd.DataFrame()
        return fig, data.to_json()

    investment_start_date = dt.datetime.strptime(investment_start_date, '%Y-%m-%d')
    # Analyse assets over a window prior to the start date
    start_date = investment_start_date - dt.timedelta(days=window * 365)
    # Evaluate the investment over one year after the start date
    end_date = investment_start_date + dt.timedelta(days=365)
    # Sort tickers in alphabetical order
    tickers = sorted(tickers)

    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        # Further work is needed to handle the exception
        raise PreventUpdate

    analysis_returns = data[start_date:investment_start_date].pct_change().dropna()
    mean_returns = analysis_returns.mean() * 252
    cov_matrix = analysis_returns.cov() * 252 if len(tickers) > 1 else analysis_returns.var() * 252
    tickers_df = pd.DataFrame({'Return': analysis_returns.mean() * 252, 'Risk': analysis_returns.std() * np.sqrt(252)},
                              index=tickers).rename_axis('Ticker')

    if len(tickers) > 1:
        # Minimum variance line
        mu, sigma_mu = minimum_variance_line(mean_returns, cov_matrix)
        # Minimum variance portfolio
        min_var_mu, min_var_sigma = minimum_variance_portfolio(mean_returns, cov_matrix)
        # Market portfolio
        market_mu, market_sigma = market_portfolio(mean_returns, cov_matrix, riskFreeRate)

    fig = px.scatter(tickers_df, x='Risk', y='Return', text=tickers_df.index)

    if len(tickers) > 1:
        fig.add_scatter(x=sigma_mu, y=mu, mode='lines', line=dict(color='black', dash='dash'),
                        name='Minimum variance line')
        fig.add_trace(go.Scatter(x=[min_var_sigma], y=[min_var_mu], mode='markers',
                                 marker=dict(size=10, color='black'), showlegend=False,
                                 name='Minimum variance portfolio', text='Minimum variance portfolio'))
        fig.add_trace(go.Scatter(x=[market_sigma], y=[market_mu], mode='markers',
                                 marker=dict(size=10, color='red'), showlegend=False,
                                 name='Market portfolio', text='Market portfolio'))
    fig.update_traces(textposition='top center').update_layout(transition_duration=500, title='Asset selection')

    return fig, data.to_json()


@app.callback(
    [Output('markowitz-graph', 'figure', allow_duplicate=True),
     Output('store-portfolios', 'data'),
     Output('ticker-dropdown', 'disabled'),
     Output('generate-button', 'n_clicks')],
    [Input('store-data', 'data'),
     Input('n-portfolios', 'value'),
     Input('start-date', 'date'),
     Input('generate-button', 'n_clicks'),
     ],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def mc_allocation(data, n_portfolios, investment_start_date, n_clicks):
    if not n_clicks:
        raise PreventUpdate

    if not data:
        raise PreventUpdate

    data = pd.read_json(StringIO(data))

    # Check if only one asset is selected by checking if data is a Series
    if isinstance(data, pd.Series):
        raise PreventUpdate
    analysis_returns = data[:investment_start_date].pct_change().dropna()

    if analysis_returns.empty:
        fig = go.Figure().update_layout(transition_duration=500)
        n_clicks = None
        mc_portfolios = pd.DataFrame()
        isDisabled = True
        return fig, mc_portfolios.to_json(), isDisabled, n_clicks

    tickers = analysis_returns.columns
    tickers_df = pd.DataFrame({'Return': analysis_returns.mean()*252, 'Risk': analysis_returns.std()*np.sqrt(252)}, index=tickers).rename_axis('Ticker')
    
    mc_portfolios = generate_portfolios(analysis_returns, n_portfolios)
    fig = px.scatter(mc_portfolios, x='Risk', y='Return', color='Sharpe Ratio', hover_data={**{ticker +' weight': ':.2f' for ticker in tickers}, **{'Return': ':.2f', 'Risk': ':.2f', 'Sharpe Ratio': ':.2f'}}, opacity=0.5,)
    fig.add_scatter(x=tickers_df['Risk'], y=tickers_df['Return'], mode='markers', marker=dict(size=7.5, color='black',),showlegend=False, name='Tickers', text = [f'<b>{index}</b> <br>Standard deviation: {vol:.2f}<br>Expected return: {ret:.2f}' for index, vol, ret in zip(tickers_df.index, tickers_df['Risk'], tickers_df['Return'])],hoverinfo='text')
    if len(mc_portfolios) <= 1000:
        fig.update_layout(transition_duration=500)

    fig.update_layout(title='Monte Carlo Simulation')

    n_clicks = None

    isDisabled = True

    return fig, mc_portfolios.to_json(), isDisabled, n_clicks


@app.callback(
    Output('portfolio-value', 'figure'),
    [Input('ticker-dropdown', 'value'),
    Input('store-data', 'data'),
    Input('store-portfolios', 'data'),
    Input('markowitz-graph', 'clickData'),
    Input('markowitz-graph', 'hoverData'),
    Input('initial-investment', 'value'),
    Input('start-date', 'date'),],
)
def plot_portfolio(tickers, data, mcPortfolios, clickData, hoverData, initial_investment, investment_start_date):
    if not clickData and not hoverData:
        raise PreventUpdate
    
    data = pd.read_json(StringIO(data)) if len(tickers) > 1 else pd.read_json(StringIO(data), typ='series')
    outOfSampleData = data[investment_start_date:]
    ylims = [((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.min()).min(), ((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.max()).max()]
    fig = go.Figure()

    if not mcPortfolios:
        if clickData:
            curveNumber = clickData['points'][0]['curveNumber']
            if curveNumber != 0:
                raise PreventUpdate
            index = clickData['points'][0]['pointNumber']
            asset_value = evaluate_asset(tickers, index, outOfSampleData, initial_investment)
            fig = px.line(asset_value).update_traces(line_color='black',)

        if hoverData:
            curveNumber = hoverData['points'][0]['curveNumber']
            if curveNumber != 0:
                raise PreventUpdate
            index = hoverData['points'][0]['pointNumber']
            asset_value = evaluate_asset(tickers, index, outOfSampleData, initial_investment)
            fig.add_trace(go.Scatter(x=asset_value.index, y=asset_value, mode='lines', opacity=0.3, line=dict(color='black')))

        fig.update_yaxes(range=ylims).update_layout(showlegend=False, title='Portfolio value')
        # else:
        #     raise PreventUpdate
    else:
        mcPortfolios = pd.read_json(StringIO(mcPortfolios))

        if clickData:
            index = clickData['points'][0]['pointNumber']
            curveNumber = clickData['points'][0]['curveNumber']
            if curveNumber == 0:
                portfolio_value = evaluate_portfolio(mcPortfolios, index, outOfSampleData, initial_investment)
                fig = px.line(portfolio_value)
            if curveNumber == 1:
                asset_value = evaluate_asset(tickers, index, outOfSampleData, initial_investment)
                fig = px.line(asset_value).update_traces(line_color='black')

        
        if hoverData:
            index = hoverData['points'][0]['pointNumber']
            curveNumber = hoverData['points'][0]['curveNumber']
            if curveNumber == 0:
                portfolio_value = evaluate_portfolio(mcPortfolios, index, outOfSampleData, initial_investment)
                fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', opacity=0.3))
            if curveNumber == 1:
                asset_value = evaluate_asset(tickers, index, outOfSampleData, initial_investment)
                fig.add_trace(go.Scatter(x=asset_value.index, y=asset_value, mode='lines', opacity=0.3, line=dict(color='black')))

        fig.update_yaxes(range=ylims).update_layout(showlegend=False, title='Portfolio value')
    
    return fig

# import json

# @app.callback(
#     Output('hover-data', 'children'),
#     Input('markowitz-graph', 'hoverData'))
# def hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)

# Delete before deploying
if __name__ == '__main__':
    app.run_server(debug=True,)