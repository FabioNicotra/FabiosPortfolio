import numpy as np
import pandas as pd
import datetime as dt

import plotly.express as px

import yfinance as yf

from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go
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
    returns = data.pct_change().dropna()
    ticker = tickers[index]
    nShares = initialValue/data.iloc[0][ticker]
    asset_value = nShares*data[ticker]
    return asset_value

# Get a list of symbols from FTSEMIB index
ftsemib = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')[1]
ftsemib['ICB Sector'] = ftsemib['ICB Sector'].str.extract(r'\((.*?)\)', expand=False).fillna(ftsemib['ICB Sector'])


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("CERULEAN")

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN, dbc_css])

app.layout = html.Div([
    dcc.Store(id='store-data'),
    dcc.Store(id='store-portfolios'),
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        className='dbc',
        children=[
            dcc.Tab(
                label='Asset selection',
                id='tab-1',
                className='dbc',
                children=[
                    dbc.Row([
                        dbc.Col(
                            width=3,
                            children=[
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
                                html.P('Select date range for analysis', className='dbc'),
                                dcc.DatePickerRange(
                                    id='analysis-date-picker',
                                    min_date_allowed=dt.date(2010, 1, 1),
                                    max_date_allowed=dt.date.today(),
                                    initial_visible_month=dt.date.today(),
                                    start_date=dt.date(2010, 1, 1),
                                    end_date=dt.date.today(),
                                    className='dbc'
                                ),

                                html.Br(),
                                html.P('Select start date for portfolio evaluation', className='dbc'),
                                dcc.DatePickerSingle(
                                    id='backtest-start-date',
                                    min_date_allowed=dt.date(2010, 1, 1),
                                    max_date_allowed=dt.date.today(),
                                    initial_visible_month=dt.date.today(),
                                    date=dt.date.today() - dt.timedelta(days=365),
                                    className='dbc'
                                ),
                            ]
                        ),
                        
                        dbc.Col(
                            dbc.Card(
                                dcc.Graph(id='markowitz-graph')
                            ),
                        )
                    ]),

                ]
            ),
            dcc.Tab(
                id='tab-2',
                label='Monte Carlo Allocation',
                children=[
                    dbc.Row([
                        dbc.Col(width=2,
                            children=dbc.Row([
                                html.P('Number of samples', className='dbc'),
                                dbc.Input(id='n-portfolios', value=1000, type='number', className='dbc'),
                                
                                dbc.Button('Generate',
                                    id='generate-button',
                                    n_clicks=0,
                                    className='dbc'),

                                html.Br(),

                                html.P('Initial investment', className='dbc'),
                                dbc.Input(id='initial-investment', value=100, type='number', className='dbc'),
                            ])),
                        dbc.Col([
                            dcc.Graph(id='mc-portfolios', 
                                      clear_on_unhover=True
                                      )
                        ], width=5),
                        dbc.Col([
                            dcc.Graph(id='portfolio-value',
                                    )
                        ], width=5),
                    ])
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
])


# Download data and plot mean-variance graph for selected assets
@app.callback(
    [Output('markowitz-graph', 'figure'),
    Output('mc-portfolios', 'figure'),
    Output('store-data', 'data')],
    [Input('ticker-dropdown', 'value'),
    Input('analysis-date-picker', 'start_date'),
    Input('analysis-date-picker', 'end_date'),
    Input('backtest-start-date', 'date')],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def select_assets(tickers, start_date, end_date, backtest_start_date):
    if not tickers:
        fig = go.Figure().update_xaxes(title='Risk', range=[0, 0.5]).update_yaxes(title='Return', range=[0, 0.4]).update_layout(transition_duration=500)
        data = pd.DataFrame()
        return fig, fig, data.to_json()

    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
    backtest_start_date = dt.datetime.strptime(backtest_start_date, '%Y-%m-%d')
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, )['Adj Close']
    except Exception as e:
        raise PreventUpdate

    analysis_returns = data[start_date:backtest_start_date].pct_change().dropna()
    tickers_df = pd.DataFrame({'Return': analysis_returns.mean()*252, 'Risk': analysis_returns.std()*np.sqrt(252)}, index=tickers).rename_axis('Ticker')

    fig = px.scatter(tickers_df, x='Risk', y='Return', text=tickers_df.index)
    fig.update_traces(textposition='top center').update_layout(transition_duration=500, title='Asset selection')

    return fig, fig, data.to_json()

@app.callback(
    [Output('mc-portfolios', 'figure', allow_duplicate=True),
    Output('store-portfolios', 'data'),
    Output('ticker-dropdown', 'disabled'),
    Output('generate-button', 'n_clicks')],
    [Input('store-data', 'data'),
    Input('n-portfolios', 'value'),
    Input('analysis-date-picker', 'start_date'),
    Input('backtest-start-date', 'date'),
    Input('generate-button', 'n_clicks'),
    ],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def mc_allocation(data, n_portfolios, analysis_start_date, analysis_end_date, n_clicks):
    if not n_clicks:
        raise PreventUpdate
    
    if not data:
        raise PreventUpdate

    data = pd.read_json(data)
    analysis_returns = data[analysis_start_date:analysis_end_date].pct_change().dropna()

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
    [Input('store-data', 'data'),
    Input('store-portfolios', 'data'),
    Input('mc-portfolios', 'clickData'),
    Input('mc-portfolios', 'hoverData'),
    Input('initial-investment', 'value'),
    Input('backtest-start-date', 'date'),
    Input('analysis-date-picker', 'end_date'),],
)
def plot_portfolio(data, mcPortfolios, clickData, hoverData, initial_investment, start_date, end_date):
    if not clickData and not hoverData:
        raise PreventUpdate


    mcPortfolios = pd.read_json(mcPortfolios)
    data = pd.read_json(data)
    outOfSampleData = data[start_date:end_date]
    tickers = data.columns
    ylims = [((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.min()).min(), ((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.max()).max()]
    fig = go.Figure()

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

    fig.update_yaxes(range=ylims).update_layout(showlegend=False, transition_duration=10, title='Portfolio value')
    
    return fig
