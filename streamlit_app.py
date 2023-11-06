import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import streamlit as st
from plotly import graph_objects as go


st.set_page_config(layout="wide")
path_to_file = 'Transactions 2023-11-03.csv'
red_rgb = 'rgb(255, 75, 75)'
red_hex = '#FF4B4B'
grey_rgb = 'rgb(49, 51, 63)'
grey_hex = '#31333F'
light_green_hex = '#A7E5D6'
dark_green_hex = '#65D2B7'

# Read data
@st.cache_data
def read_data(path_to_file):
    df= pd.read_csv(path_to_file, parse_dates=['Date'])
    df = df.iloc[:-1]
    # Process date
    df['Date'] = [i.split(' ')[0] for i in df['Date']]
    df['Date'] = [dt.strptime(i, '%m/%d/%Y') for i in df['Date']]
    df = df.set_index('Date')
    # Clean currency columns
    curreny_columns = ['Price', 'Fees & Comm', 'Amount']
    for i in curreny_columns:
        df[i] = pd.to_numeric(df[i].replace('[^0-9\.-]', '', regex=True))
    # Split symbol column
    symbol_list = []
    exp_list = []
    strike_list = []
    action_list = []
    for i in df['Symbol']:
        try:
            split = i.split(' ')
            if len(split) == 1:
                symbol_list.append(i)
                exp_list.append(np.nan)
                strike_list.append(np.nan)
                
            else:
                symbol_list.append(split[0])
                exp_list.append(split[1])
                strike_list.append(split[2])
                
        except:
            symbol_list.append(np.nan)
            exp_list.append(np.nan)
            strike_list.append(np.nan)
            
    df['Symbol'] = symbol_list
    df['Exp. Date'] = exp_list
    df['Strike'] = strike_list
    
    for i,x in enumerate(df['Exp. Date']):
        if type(x) == str:
            df['Exp. Date'].iloc[i] = dt.strptime(x, '%m/%d/%Y')
    
    return df

#df = read_data(path_to_file)

@st.cache_data
def get_years(dataframe):
    years = dataframe.groupby(pd.Grouper(freq='Y')).size().index.tolist()
    years = [i.to_pydatetime() for i in years]
    #years = [i.strftime('%Y') for i in years]
    return years

    
@st.cache_data
def total_premiums(dataframe, min_year, max_year):
    # Used for area chart
    query = dataframe.query('Action == "Sell to Open" and index.dt.year >= @min_year and index.dt.year <= @max_year')
    # Used for metrics
    total_prem = query['Amount'].sum().round(2)
    # Used for bar charts
    symbol_prem_totals = query.groupby('Symbol')['Amount'].sum().round(2).reset_index().sort_values('Amount',ascending=True)
    # Add a new column '% of Total' that represents the percentage of each value in column 'A'
    symbol_prem_totals['Percent of total'] = symbol_prem_totals['Amount'] / symbol_prem_totals['Amount'].sum()
    # Used for bar charts
    symbol_prem_monthly = query.groupby('Symbol').resample('M')['Amount'].sum().reset_index()
    symbol_prem_monthly['Percent of total'] = symbol_prem_monthly['Amount'] / symbol_prem_monthly['Amount'].sum()
    # Used for metrics
    monthly_prem_avg = query.resample('M')['Amount'].sum().mean()
    # Used for bar charts
    symbol_monthly_avg = symbol_prem_monthly.groupby('Symbol')['Amount'].mean().round(2).reset_index().sort_values('Amount',ascending=True)
    symbol_monthly_avg['Percent of total'] = symbol_monthly_avg['Amount'] / symbol_monthly_avg['Amount'].sum()
    #for timeline chart
    chart_query = query.copy()
    chart_query.reset_index(inplace=True)
    chart_query = chart_query.sort_values('Exp. Date', ascending=True)
    chart_query['Date'] = chart_query['Date'].apply(lambda x: str(x).split(' ')[0]) 
    chart_query['Exp. Date'] = chart_query['Exp. Date'].apply(lambda x: str(x).split(' ')[0]) 
    unique_symbols = chart_query['Symbol'].unique()
    data = {"line_x": [], "line_y": [], "sold":[], 'expired':[], "symbols": [], 'size': [], 'price': [], 'quantity': [], 'amount':[], 'text':[]}
    for sym in unique_symbols:
        tmp = chart_query.query('Symbol == @sym').reset_index(drop=True)
        for i, x in enumerate(range(len(tmp))):
            data["line_x"].extend(
                [
                    tmp['Date'].values[i],
                    tmp['Exp. Date'].values[i],
                    None,
                ]
            )
            data["line_y"].extend([sym, sym, None])
            data['symbols'].extend([sym])
            data['sold'].extend([tmp['Date'][i]])
            data['expired'].extend([tmp['Exp. Date'][i]])
            data['price'].extend([tmp['Price'][i]])
            data['quantity'].extend([tmp['Quantity'][i]])
            data['amount'].extend([tmp['Amount'][i]])
            data["size"].extend([tmp['Amount'][i]/10])
            quantitiy = tmp['Quantity'][i]
            amount = tmp['Amount'][i]
            price = tmp['Price'][i]
            strike = tmp['Strike'][i]
            data['text'].extend([f'Strike: ${strike}<br>Quantity: {quantitiy}<br>Price: ${price}<br>Amount: ${amount}'])
        
    return total_prem, monthly_prem_avg, symbol_prem_totals, symbol_monthly_avg, query, data

@st.cache_data
def total_contracts(dataframe, min_year, max_year):
    # Used for area chart
    contract_query = dataframe.query('Action == "Sell to Open" and index.dt.year >= @min_year and index.dt.year <= @max_year')
    # Used for metrics
    sold_contracts = contract_query['Quantity'].sum()
    symbol_contracts = contract_query.groupby('Symbol')['Quantity'].sum().reset_index().sort_values('Quantity',ascending=True)
    symbol_contracts['Percent of total'] = symbol_contracts['Quantity'] / symbol_contracts['Quantity'].sum()
    assigned_query = dataframe.query('Action == "Assigned" and index.dt.year >= @min_year and index.dt.year <= @max_year')
    assigned_contracts = assigned_query['Quantity'].sum()
    assigned_symbols = assigned_query.groupby('Symbol')['Quantity'].sum().reset_index().sort_values('Quantity',ascending=True)
    assigned_symbols['Percent of total'] = assigned_symbols['Quantity'] / assigned_symbols['Quantity'].sum()
    return sold_contracts, assigned_contracts, symbol_contracts, assigned_symbols


@st.cache_data
def monthly_premium(dataframe, min_year, max_year):
    monthly_prem = df.query('Action == "Sell to Open" and index.dt.year >= @min_year and index.dt.year <= @max_year')['Amount'].resample('M').sum()
    calls = df.query('Action == "Sell to Open"').reset_index()
    calls['Date'] = pd.to_datetime(calls['Date'])
    calls['Exp. Date'] = pd.to_datetime(calls['Exp. Date'])
    calls['Days Out'] = (calls['Exp. Date'] - calls['Date']).dt.days
    calls_monthly = calls.copy()
    calls_monthly['Date'] = calls_monthly['Date'].dt.strftime('%Y-%m')
    calls_monthly = calls_monthly.groupby(['Date','Symbol']).agg({'Days Out': 'mean', 'Quantity': 'sum', 'Amount': 'sum', 'Exp. Date': 'max'}).reset_index().set_index('Date')
    return monthly_prem

def discrete_colors(dataframe, light_color, dark_color):
    colors = [light_color] * len(dataframe)
    colors[-1] = dark_color
    return colors

def symbol_bars(x, y, custom_data, prefix):
            fig = go.Figure(go.Bar(
                    x=x,
                    y=y,
                    orientation='h',
                    texttemplate="%{y} "+prefix+"%{x}",
                    textposition="auto",
                    customdata=custom_data,
                    hovertemplate=
                        "<b>%{y}</b><br><br>" +
                        "Premiums sold: %{x:prefix,.0f}<br>" +
                        "% of total: %{customdata:.0%}<br>" +
                        "<extra></extra>",
                    #width=0.4
                ))
            fig.update_layout(
                yaxis_visible=False, 
                yaxis_showticklabels=False,
                xaxis_visible=False, 
                xaxis_showticklabels=False,
                #hovermode="x unified",
                margin=dict(l=0, r=30, t=0, b=0),
                autosize=False,
                height=100,
                bargap=0.1
            )
            fig.update_traces(marker_color=colors)
            return fig

st.title('Schwab Options History Analysis')

uploaded_file = st.sidebar.file_uploader("Upload Schwab transactions history", type="csv")

if uploaded_file is not None:
    df = read_data(uploaded_file)

    if 'df' not in st.session_state: 
        st.session_state.df = df

    years = get_years(st.session_state.df)
    min_year, max_year = st.sidebar.slider(
        "Year selector",
        value=[years[-1], years[-1]],
        step=datetime.timedelta(days=365),
        min_value=years[0],
        max_value=years[-1],
        format="YYYY",    
    )

    min_year = int(min_year.strftime('%Y'))
    max_year = int(max_year.strftime('%Y'))

    total_prem, monthly_prem_avg, symbol_prem_totals, symbol_monthly_avg, query, data = total_premiums(st.session_state.df, min_year, max_year)
    sold_contracts, assigned_contracts, symbol_contracts, assigned_symbols = total_contracts(st.session_state.df, min_year, max_year)

    body = st.container()

    with body:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            with st.container():
                st.metric(label="Total Premiums", value=f"${total_prem:,.0f}")
                colors = discrete_colors(symbol_prem_totals, light_green_hex, dark_green_hex)
                fig = symbol_bars(symbol_prem_totals['Amount'], symbol_prem_totals['Symbol'], symbol_prem_totals['Percent of total'], '$')
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

        with col2:
            with st.container():
                st.metric(label="Avg Monthly Premiums", value=f"${monthly_prem_avg:,.0f}")
                colors = discrete_colors(symbol_monthly_avg, light_green_hex, dark_green_hex)
                fig = symbol_bars(symbol_monthly_avg['Amount'], symbol_monthly_avg['Symbol'], symbol_monthly_avg['Percent of total'], '$')
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
        
        with col3:
            with st.container():
                st.metric(label="Total Contracts Sold", value=f"{sold_contracts:,.0f}")
                colors = discrete_colors(symbol_contracts, light_green_hex, dark_green_hex)
                fig = symbol_bars(symbol_contracts['Quantity'], symbol_contracts['Symbol'], symbol_contracts['Percent of total'], '')
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
        
        with col4:
            with st.container():
                st.metric(label="Total Contracts Assigned", value=f"{assigned_contracts:,.0f}")
                colors = discrete_colors(symbol_contracts, light_green_hex, dark_green_hex)
                fig = symbol_bars(assigned_symbols['Quantity'], assigned_symbols['Symbol'], assigned_symbols['Percent of total'], '')
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

        st.divider()
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=pd.to_datetime(pd.Series(data["line_x"]), format='%Y-%m-%d'),
                    y=data["line_y"],
                    mode="lines",
                    showlegend=False,
                    marker=dict(
                        color="grey"
                    ),
                ),
                go.Scatter(
                    x=pd.to_datetime(pd.Series(data["sold"]), format='%Y-%m-%d'),
                    y=data["symbols"],
                    mode="markers",
                    name="Sold Date",
                    marker=dict(
                        color=dark_green_hex,
                        size=data['size']
                    ),
                customdata=data['text'],
                    hovertemplate=
                        "<b>%{y}</b><br><br>" +
                        "%{customdata}<br>" +
                        "<extra></extra>",
                    
                ),
                go.Scatter(
                    x=pd.to_datetime(pd.Series(data["expired"]), format='%Y-%m-%d'),
                    y=data["symbols"],
                    mode="markers",
                    name="Exp. Date",
                    marker=dict(
                        color=red_hex,
                        size=data['size']
                    ),
                    customdata=data['text'],
                    hovertemplate=
                        "<b>%{y}</b><br><br>" +
                        "%{customdata}<br>" +
                        "<extra></extra>",
                ),
            ]
        )
        fig.add_vline(x=dt.today(), line_width=1, line_dash="dash", line_color=red_hex)
        fig.update_layout(title='Trade activity')
        st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

