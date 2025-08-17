"""
Dash dashboard for stock prediction visualization
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard"),
    
    html.Div([
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-selector',
            options=[
                {'label': 'Reliance', 'value': 'RELIANCE.NS'},
                {'label': 'TCS', 'value': 'TCS.NS'},
                {'label': 'HDFC Bank', 'value': 'HDFCBANK.NS'},
                {'label': 'Infosys', 'value': 'INFY.NS'}
            ],
            value='RELIANCE.NS'
        )
    ], style={'width': '30%', 'margin': '10px'}),
    
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=datetime(2010, 1, 1),
            max_date_allowed=datetime.now(),
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
    ], style={'margin': '10px'}),
    
    html.Button('Predict', id='predict-button', n_clicks=0),
    
    dcc.Graph(id='stock-graph'),
    
    html.Div([
        html.H3("Prediction Details"),
        html.Div(id='prediction-details')
    ], style={'margin': '20px'})
])

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('prediction-details', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('stock-selector', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_graph(n_clicks, stock, start_date, end_date):
    if n_clicks == 0:
        return dash.no_update
    
    # Load historical data
    df = pd.read_csv(f'data/{stock}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df.loc[mask]
    
    # Make prediction request
    try:
        response = requests.post(
            'http://localhost:8000/predict',
            json={
                'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist()
            }
        )
        prediction_data = response.json()
        predicted_price = prediction_data['prediction']
        confidence = prediction_data['confidence']
        
        # Create prediction point
        next_day = pd.to_datetime(end_date) + timedelta(days=1)
        
        # Create the graph
        figure = {
            'data': [
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    name='Historical',
                    mode='lines'
                ),
                go.Scatter(
                    x=[df['Date'].iloc[-1], next_day],
                    y=[df['Close'].iloc[-1], predicted_price],
                    name='Prediction',
                    mode='lines',
                    line=dict(dash='dash')
                )
            ],
            'layout': go.Layout(
                title=f'{stock} Stock Price',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                hovermode='closest'
            )
        }
        
        # Create prediction details
        details = html.Div([
            html.P(f"Predicted Price: ₹{predicted_price:.2f}"),
            html.P(f"Confidence: {confidence:.2%}"),
            html.P(f"Prediction Date: {next_day.strftime('%Y-%m-%d')}")
        ])
        
        return figure, details
        
    except Exception as e:
        return dash.no_update, html.Div(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True) 