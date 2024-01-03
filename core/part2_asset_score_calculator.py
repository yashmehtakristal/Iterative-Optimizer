# Importing various libraries
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import scale
import ffn
import io
import streamlit as st
import base64 
#from cryptography.fernet import Fernet

@st.cache_data(show_spinner = False)
def reading_returns_file():
    '''
    Reads adjusted_asset_returns.xlsx (named as returns) and does certain filtering by date
    
    Input: None
    Output: Returns dataframe
    '''

    # Read excel file
    returns = pd.read_excel('input_data/adjusted_asset_returns.xlsx')

    return returns

@st.cache_data(show_spinner = False)
def returns_preprocessing(returns, start_date, end_date):
    '''
    Reads adjusted_asset_returns.xlsx (named as returns) and does certain filtering by date
    
    Input: None
    Output: Returns dataframe
    '''

    returns['Date']= pd.to_datetime(returns['Date'])
    returns.set_index('Date', inplace=True)

    ## TODO - Need to change this to Streamlit code
    #Selecting returns history post mar'2019
    # start_date = pd.to_datetime('2019-03-29')
    # end_date = pd.to_datetime('2023-08-31')

    # This will select only rows corresponding to the start and end date
    returns = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
    returns = returns.sort_values(by = ['Date'])

    # Sorts by column named "Date" in ascending order
    returns = returns.loc[:,returns.notna().all(axis=0)]

    return returns


#Function for generating drifting series starting from portfolio weight
# asset_returns - List of asset returns
# asset_weight - Initial weight of asset in series
@st.cache_data(show_spinner = False)
def gen_drift_series(asset_returns, asset_weight):
    
    # Length of the asset returns
    num_returns = len(asset_returns)
    
    print(num_returns)
    
    # initializes an empty list named drifting_series with a length of num_returns + 1
    drifting_series = [0] * (num_returns + 1)
    
    # Initializing first element as asset weight
    drifting_series[0] = asset_weight
    
    # Calculating compounded returns over time period
    for i in range(num_returns):
        drifting_series[i+1] = drifting_series[i]*(asset_returns[i]+1)
    
    return drifting_series

@st.cache_data(show_spinner = False)
def obtain_respective_asset_value(kristal_mappings, kristal_name_selected):

    # Use boolean indexing to filter the DataFrame for the selected kristal_name
    selected_row = kristal_mappings[kristal_mappings['kristal_name'] == kristal_name_selected]

    # Access the "Asset" column value for the selected row
    asset_selected = selected_row['Asset'].values[0]

    return asset_selected


@st.cache_data(show_spinner = False)
def read_kristal_mappings():
    '''
    Reads Asset Index mapping excel file as kristal_mappings after applying certain conditions

    Input: None
    Output: kristal_mappings dataframe, asset_universe, adjusted_returns_universe
    '''

    kristal_mappings = pd.read_excel('input_data/Asset Index mapping.xlsx')

    return kristal_mappings


@st.cache_data(show_spinner = False)
def kristal_mapping_manipulation(kristal_mappings, returns):

    # notna() checks for each row in 'kristalid' column if value is not null or missing
    # .loc[] selects only rows where 'kristalid' column is not null
    # Resets the row indices to default integer indices
    kristal_mappings = kristal_mappings.loc[kristal_mappings['kristalid'].notna()].reset_index()

    # Selecting only particular columns and renaming them
    kristal_mappings = kristal_mappings[['kristalid', 'kristal_name', 'kristal_sub_type', 'Asset', 'Index']]
    kristal_mappings.columns = ['kristal_id', 'kristal_name', 'kristal_sub_type', 'kristal_ticker', 'kristal_index']

    # Extracting unique values from kristal_ticker (Asset) column
    asset_universe = list(set(kristal_mappings['kristal_ticker']))

    # Get the column headers (first row) of the returns dataframe file
    adjusted_returns_universe = list(set(returns.columns.values))

    # Define unique list which now contains only the unique asset identifiers that are present in both the 'kristal_ticker'
    # column of kristal_mappings and the column names of the returns DataFrame.
    asset_universe = list(set(asset_universe) & set(adjusted_returns_universe))

    return kristal_mappings, asset_universe, adjusted_returns_universe


@st.cache_data(show_spinner = False)
def read_portfolios():
    '''
    Reads portfolio excel file and does preliminary manipulation
    
    Input: kristal_mappings dataframe
    Output: portfolio1 dataframe
    '''

    #Reading input file - client portfolios
    portfolios = pd.read_excel('input_data/client_data.xlsx', 'Portfolio')

    return portfolios


@st.cache_data(show_spinner = False)
def read_portfolios_for_particular_client(portfolios, client_id):

    # Only return the particular rows where client_id column has particular client_id value and reset the index
    portfolio1 = portfolios.loc[portfolios['client_id'] == client_id].reset_index()

    # Select all rows
    # as well as all columns except the first column (starting from second column onwards)
    portfolio1 = portfolio1.iloc[: , 1:]


    return portfolio1

@st.cache_data(show_spinner = False)
def portfolio_manipulation(portfolio1, returns, kristal_mappings, asset_universe, asset_ticker):

    # Getting mappings data for the portfolio assets
    # merges portfolio1 dataframe with kristal_mappings on kristal_Id
    # Left join - All rows from portfolio1 are retained, but matching rows in kristal_mappings are added.
    portfolio1 = portfolio1.merge(kristal_mappings, on='kristal_id', how='left')

    # 'portfolio1' is filtered to keep only rows where kristal_ticker is in asset_universe 
    # and then index is resetted
    portfolio1 = portfolio1[portfolio1['kristal_ticker'].isin(asset_universe)].reset_index()

    # Calculate total portfolio value by summing up values in 'AUM in USD' column and storing it in portfolio_value
    portfolio_value = portfolio1['AUM in USD'].sum()

    # Calculate the weight of each asset in the portfolio
    portfolio1['weight'] = portfolio1['AUM in USD']/portfolio_value


    # Obtaining number of assets in portfolio by looking at the number of rows in portfolio1
    num_assets = portfolio1.shape[0]

    # Length of 'returns' dataframe
    num_returns = len(returns)

    # Initializing list of port_prices
    port_prices = [0]*(num_returns+1)

    # For loop iterating num_assets times
    # This will find the returns of each asset in portfolio and essentially sum it 
    for i in range(num_assets):
        #print(portfolio1['asset'][i])

        # Call function gen_drift_series with returns of particular portfolio and their weights
        # Then, add the series of price changes generated by gen_drift_series
        port_prices = np.add(port_prices, gen_drift_series(returns[portfolio1['kristal_ticker'][i]], portfolio1['weight'][i]))
    
    
    # Calculating the length of the portfolio prices
    len_prices = len(port_prices)

    # This calculates the % change (returns) of the index_drift_series
    port_returns = pd.DataFrame(port_prices).pct_change()

    # Remove the last row (as it might possibly have NaN values) and resetting index
    port_returns = port_returns.tail(-1).reset_index()

    # Selects all rows and all columns starting from the second column 
    port_returns = port_returns.iloc[: , 1:]

    # Storing end_price variable
    end_price = port_prices[len_prices-1]

    # Calculating CAGR, Volatility and Portfolio Sharpe on Portfolio Prices
    cagr = np.power(end_price,(12/(len_prices-1)))-1
    vol = np.std(port_returns, ddof=1)*np.sqrt(12)
    port_sharpe = cagr/vol


    #Generating stats with 10% new asset in the portfolio

    # Calling the gen_drift_series function for portfolio returns (with 90% weight to old portfolio)
    portfolio_drift_series = gen_drift_series(port_returns[port_returns.columns[0]], 0.9)

    # Test
    # returns_columns = returns.columns
    # test_selectbox = st.selectbox(label = "Please select the client id from dropdown", options = returns_columns, index = None, key = "test_select_box", help = "select_client_id", placeholder="Choose particular client id", disabled = False, label_visibility = "visible")
    # st.write(returns)

    # Calling the gen_drift_series function with 10% weight to new asset in portfolio
    asset_drift_series = gen_drift_series(returns[asset_ticker], 0.1)

    # Adding the two portfolio returns
    new_port_prices = np.add(portfolio_drift_series, asset_drift_series)

    #Using both the drift series to generate weights column - how weights change over time 
    new_asset_weights = [0]*(len(port_prices))

    # Calculating the new asset weights
    for i in range(len_prices):
        new_asset_weights[i] = asset_drift_series[i]/(portfolio_drift_series[i] + asset_drift_series[i])

    len_new_port_prices = len(new_port_prices)

    # This calculates the % change (returns) of the new_port_prices
    new_port_returns = pd.DataFrame(new_port_prices).pct_change()

    # Remove the last row (as it might possibly have NaN values) and resetting index
    new_port_returns = new_port_returns.tail(-1).reset_index()

    # Selects all rows and all columns starting from the second column 
    new_port_returns = new_port_returns.iloc[: , 1:]

    # Retrieves last value (ending price) in new_port_returns
    new_end_price = new_port_prices[len_new_port_prices-1]

    # Calculate new metrics: CAGR, Vol and Sharpe
    new_cagr = np.power(new_end_price,(12/(len_new_port_prices-1)))-1
    new_vol = np.std(new_port_returns, ddof=1)*np.sqrt(12)
    new_port_sharpe = new_cagr/new_vol

    # Calculates change of sharpe, cagr and volatility with just portfolio
    sharpe_change = (new_port_sharpe - port_sharpe)/port_sharpe
    cagr_change = (new_cagr - cagr)/cagr
    vol_change = (vol - new_vol)/vol

    #Making dates column for prices

    current_date = pd.to_datetime('2019-03-01')

    dates = ['']*len_prices
    dates[0] = current_date.strftime('%d/%m/%Y')

    # Adding a month forr each subsequent date
    for i in range(len_prices-1):
        current_date = current_date + relativedelta(months=1)
        dates[i+1] = current_date.strftime('%d/%m/%Y')     

    return dates, port_prices, new_port_prices, new_asset_weights, cagr, new_cagr, len_prices, len_new_port_prices, port_returns, new_port_returns, end_price, new_end_price

@st.cache_data(show_spinner = False)
def create_prices_dataframe(dates, port_prices, asset_ticker, new_port_prices, new_asset_weights, returns):

    #Dataframe for Prices Sheet
    prices_df = pd.DataFrame()
    prices_df['Date'] = pd.Series(dates)
    prices_df['Port Prices'] = port_prices
    asset_prices = gen_drift_series(returns[asset_ticker], 1)
    prices_df['New Port Prices'] = new_port_prices
    prices_df['Asset Prices'] = asset_prices
    prices_df['Asset Weights'] = new_asset_weights
    prices_df.set_index('Date', inplace=True)

    return prices_df


@st.cache_data(show_spinner = False)
def create_returns_dataframe(returns, port_returns, new_port_returns, asset_ticker):

    #Dataframe for Returns Sheet 
    returns_df = pd.DataFrame()
    returns_df['Date'] = returns.index
    returns_df['Port Returns'] = port_returns
    returns_df['New Port Returns'] = new_port_returns
    returns_df.set_index('Date', inplace=True)
    returns_df['Asset Returns'] = returns[asset_ticker]

    return returns_df

@st.cache_data(show_spinner = False)
def create_stats_dataframe(cagr, new_cagr, end_price, new_end_price, port_returns, len_prices, len_new_port_prices, returns_df, prices_df, new_port_returns, port_prices, new_port_prices, dates, returns):

    #Stats DataFrame
    stats_df = pd.DataFrame(columns=['stat', 'old_port', 'new_port', 'ref_date'])

    # Populating first row with CAGR data 
    stats_df.loc[0] = ['CAGR', cagr*100, new_cagr*100, np.nan]

    # Subtracting the end_price by 1
    total_ret = end_price - 1

    # Subtracting the new_end_price by 1
    new_total_ret = new_end_price-1

    # Populating second row with Return data 
    stats_df.loc[1] = ['Complete Return', total_ret*100, new_total_ret*100, np.nan]

    # Populating third row with difference between max return and minimum return
    stats_df.loc[2] = ['Max Return - Min Return', (port_returns[0].max()-port_returns[0].min())*100, 
                    (new_port_returns[0].max()-new_port_returns[0].min())*100, np.nan]

    #Creating 12 month rolling returns 
    rolling_rets = [0]*(len_prices - 12)

    for i in range(len_prices - 12):
        rolling_rets[i] = port_prices[i+12]/port_prices[i]-1

    new_rolling_rets = [0]*(len_new_port_prices-12)
    rolling_rets_diff = [0]*(len_new_port_prices-12)


    for i in range(len_new_port_prices - 12):
        new_rolling_rets[i] = new_port_prices[i+12]/new_port_prices[i]-1
        rolling_rets_diff[i] = new_rolling_rets[i] - rolling_rets[i]

    #Max Outperformance Stat over 12 month rolling rets 
    max_rolling_ret_diff = max(rolling_rets_diff)
    max_diff_index = rolling_rets_diff.index(max_rolling_ret_diff)
    max_diff_start_dt = dates[max_diff_index]
    max_diff_end_dt = (pd.to_datetime(dates[max_diff_index], format='%d/%m/%Y') + relativedelta(months=12)).strftime('%d/%m/%Y')
    ref_dt_str = max_diff_start_dt + '-' + max_diff_end_dt
    stats_df.loc[3] = ['Max Outperformance', rolling_rets[max_diff_index]*100, new_rolling_rets[max_diff_index]*100, ref_dt_str]

    #Max Drawdowns 
    mdd = ffn.core.calc_max_drawdown(pd.Series(port_prices))
    new_mdd = ffn.core.calc_max_drawdown(pd.Series(new_port_prices))
    stats_df.loc[4] = ['Max Drawdown', mdd*100, new_mdd*100, np.nan]

    #Mean negative return 
    total_neg_ret = port_returns.where(port_returns<0).sum()[0]
    new_total_neg_ret = new_port_returns.where(new_port_returns<0).sum()[0]
    stats_df.loc[5] = ['Total Neg Return', total_neg_ret*100, new_total_neg_ret*100, np.nan]

    #Mean positive return
    total_pos_ret = port_returns.where(port_returns>0).sum()[0]
    new_total_pos_ret = new_port_returns.where(new_port_returns>0).sum()[0]
    stats_df.loc[6] = ['Total Pos Return', total_pos_ret*100, new_total_pos_ret*100, np.nan]

    #Best Port Month 
    max_port_ret = max(returns_df['Port Returns'])
    max_port_ret_index = returns_df.index[returns_df['Port Returns']==max_port_ret]
    parallel_new_port_ret = returns_df['New Port Returns'].loc[max_port_ret_index]
    stats_df.loc[7] = ['Best Port Month', max_port_ret*100, parallel_new_port_ret[0]*100, max_port_ret_index.strftime('%d/%m/%Y')[0]]

    #Worst Port Month
    min_port_ret = min(returns_df['Port Returns'])
    min_port_ret_index = returns_df.index[returns_df['Port Returns']==min_port_ret]
    parallel_new_port_ret2 = returns_df['New Port Returns'].loc[min_port_ret_index]
    stats_df.loc[8] = ['Worst Port Month', min_port_ret*100, parallel_new_port_ret2[0]*100, min_port_ret_index.strftime('%d/%m/%Y')[0]]

    #Best New Port Month 
    max_new_port_ret = max(returns_df['New Port Returns'])
    max_new_port_ret_index = returns_df.index[returns_df['New Port Returns']==max_new_port_ret]
    parallel_old_port_ret = returns_df['Port Returns'].loc[max_new_port_ret_index]
    stats_df.loc[9] = ['Best New Port Month', parallel_old_port_ret[0]*100, max_new_port_ret*100, max_new_port_ret_index.strftime('%d/%m/%Y')[0]]

    #Worst New Port Month 
    min_new_port_ret = min(returns_df['New Port Returns'])
    min_new_port_ret_index = returns_df.index[returns_df['New Port Returns']==min_new_port_ret]
    parallel_old_port_ret2 = returns_df['Port Returns'].loc[min_new_port_ret_index]
    stats_df.loc[10] = ['Worst New Port Month', parallel_old_port_ret2[0]*100, min_new_port_ret*100, min_new_port_ret_index.strftime('%d/%m/%Y')[0]]

    #Last 1 yr Sharpe 
    port_1yr_sharpe = ffn.core.calc_sharpe(returns_df['Port Returns'].tail(12), rf=0.0, nperiods=12, annualize=True)
    new_port_1yr_sharpe = ffn.core.calc_sharpe(returns_df['New Port Returns'].tail(12), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[11] = ['Last 1 yr Sharpe', port_1yr_sharpe, new_port_1yr_sharpe, np.nan]

    #Last 1 yr Sortino
    port_1yr_sortino = ffn.core.calc_sortino_ratio(returns_df['Port Returns'].tail(12), rf=0.0, nperiods=12, annualize=True)
    new_port_1yr_sortino = ffn.core.calc_sortino_ratio(returns_df['New Port Returns'].tail(12), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[12] = ['Last 1 yr Sortino', port_1yr_sortino, new_port_1yr_sortino, np.nan]

    #Last 1 yr Volatility
    port_1yr_vol = np.std(returns_df['Port Returns'].tail(12)) * np.sqrt(12)
    new_port_1yr_vol = np.std(returns_df['New Port Returns'].tail(12)) * np.sqrt(12)
    stats_df.loc[13] = ['Last 1 yr Volatility', port_1yr_vol, new_port_1yr_vol, np.nan]

    #Last 2 yrs Sharpe
    port_2yr_sharpe = ffn.core.calc_sharpe(returns_df['Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    new_port_2yr_sharpe = ffn.core.calc_sharpe(returns_df['New Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[13] = ['Last 2 yr Sharpe', port_2yr_sharpe, new_port_2yr_sharpe, np.nan]

    #Last 2 yrs Sharpe
    port_2yr_sharpe = ffn.core.calc_sharpe(returns_df['Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    new_port_2yr_sharpe = ffn.core.calc_sharpe(returns_df['New Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[14] = ['Last 2 yr Sharpe', port_2yr_sharpe, new_port_2yr_sharpe, np.nan]

    #Last 2 yrs Sortino
    port_2yr_sortino = ffn.core.calc_sortino_ratio(returns_df['Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    new_port_2yr_sortino = ffn.core.calc_sortino_ratio(returns_df['New Port Returns'].tail(24), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[15] = ['Last 2 yr Sortino', port_2yr_sortino, new_port_2yr_sortino, np.nan]

    #Last 2 yrs Volatility
    port_2yr_vol = np.std(returns_df['Port Returns'].tail(24)) * np.sqrt(12)
    new_port_2yr_vol = np.std(returns_df['New Port Returns'].tail(24)) * np.sqrt(12)
    stats_df.loc[16] = ['Last 2 yr Volatility', port_2yr_sortino, new_port_2yr_sortino, np.nan]

    #Last 3 yrs Sharpe
    port_3yr_sharpe = ffn.core.calc_sharpe(returns_df['Port Returns'].tail(36), rf=0.0, nperiods=12, annualize=True)
    new_port_3yr_sharpe = ffn.core.calc_sharpe(returns_df['New Port Returns'].tail(36), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[17] = ['Last 3 yr Sharpe', port_3yr_sharpe, new_port_3yr_sharpe, np.nan]

    #Last 3 yrs Sortino
    port_3yr_sortino = ffn.core.calc_sortino_ratio(returns_df['Port Returns'].tail(36), rf=0.0, nperiods=12, annualize=True)
    new_port_3yr_sortino = ffn.core.calc_sortino_ratio(returns_df['New Port Returns'].tail(36), rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[18] = ['Last 3 yr Sortino', port_3yr_sortino, new_port_3yr_sortino, np.nan]

    #Last 3 yrs Volatility
    port_3yr_vol = np.std(returns_df['Port Returns'].tail(24)) * np.sqrt(12)
    new_port_3yr_vol = np.std(returns_df['New Port Returns'].tail(24)) * np.sqrt(12)
    stats_df.loc[19] = ['Last 3 yr Volatility', port_3yr_vol, new_port_3yr_vol, np.nan]

    #Sharpe
    port_sharpe = ffn.core.calc_sharpe(returns_df['Port Returns'], rf=0.0, nperiods=12, annualize=True)
    new_port_sharpe = ffn.core.calc_sharpe(returns_df['New Port Returns'], rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[20] = ['Sharpe', port_sharpe, new_port_sharpe, np.nan]

    #Sortino
    port_sortino = ffn.core.calc_sortino_ratio(returns_df['Port Returns'], rf=0.0, nperiods=12, annualize=True)
    new_port_sortino = ffn.core.calc_sortino_ratio(returns_df['New Port Returns'], rf=0.0, nperiods=12, annualize=True)
    stats_df.loc[21] = ['Sortino', port_sortino, new_port_sortino, np.nan]

    #Sharpe
    port_volatility = np.std(returns_df['Port Returns']) * np.sqrt(12)
    new_port_volatility = np.std(returns_df['New Port Returns']) * np.sqrt(12)
    stats_df.loc[22] = ['Volatility', port_volatility, new_port_volatility, np.nan]

    #2019 returns 
    start_price = prices_df.loc[prices_df.index == '01/03/2019']['Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2020']['Port Prices'][0]
    port_ret_2019 = end_price/start_price-1
    start_price = prices_df.loc[prices_df.index == '01/03/2019']['New Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2020']['New Port Prices'][0]
    new_port_ret_2019 = end_price/start_price-1
    stats_df.loc[23] = ['2019 Return', port_ret_2019*100, new_port_ret_2019*100, np.nan]

    #2020 returns
    start_price = prices_df.loc[prices_df.index == '01/01/2020']['Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2021']['Port Prices'][0]
    port_ret_2020 = end_price/start_price-1
    start_price = prices_df.loc[prices_df.index == '01/01/2020']['New Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2021']['New Port Prices'][0]
    new_port_ret_2020 = end_price/start_price-1
    stats_df.loc[24] = ['2020 Return', port_ret_2020*100, new_port_ret_2020*100, np.nan]

    #2021 returns 
    start_price = prices_df.loc[prices_df.index == '01/01/2021']['Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2022']['Port Prices'][0]
    port_ret_2021 = end_price/start_price-1
    start_price = prices_df.loc[prices_df.index == '01/01/2021']['New Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2022']['New Port Prices'][0]
    new_port_ret_2021 = end_price/start_price-1
    stats_df.loc[25] = ['2021 Return', port_ret_2021*100, new_port_ret_2021*100, np.nan]

    #2022 returns 
    start_price = prices_df.loc[prices_df.index == '01/01/2022']['Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2023']['Port Prices'][0]
    port_ret_2022 = end_price/start_price-1
    start_price = prices_df.loc[prices_df.index == '01/01/2022']['New Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/01/2023']['New Port Prices'][0]
    new_port_ret_2022 = end_price/start_price-1
    stats_df.loc[26] = ['2022 Return', port_ret_2022*100, new_port_ret_2022*100, np.nan]

    #2023 returns
    start_price = prices_df.loc[prices_df.index == '01/01/2023']['Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/09/2023']['Port Prices'][0]
    port_ret_2023 = end_price/start_price-1
    start_price = prices_df.loc[prices_df.index == '01/01/2023']['New Port Prices'][0]
    end_price = prices_df.loc[prices_df.index == '01/09/2023']['New Port Prices'][0]
    new_port_ret_2023 = end_price/start_price-1
    stats_df.loc[27] = ['2023 Return', port_ret_2023*100, new_port_ret_2023*100, np.nan]

    #Num of positive return months 
    port_pos_mths = len(returns_df[returns_df['Port Returns'] >= 0])
    new_port_pos_mths = len(returns_df[returns_df['New Port Returns'] >= 0])
    stats_df.loc[28] = ['Pos Return Months', port_pos_mths, new_port_pos_mths, np.nan]

    #Num of negative return months 
    port_neg_mths = len(returns_df[returns_df['Port Returns'] < 0])
    new_port_neg_mths = len(returns_df[returns_df['New Port Returns'] < 0])
    stats_df.loc[29] = ['Neg Return Months', port_neg_mths, new_port_neg_mths, np.nan]

    #3 yr correlation with SPY
    spy_ticker = 'SPY US Equity' 
    spy_3yr_returns = returns[spy_ticker].tail(36)
    port_3yr_returns = returns_df['Port Returns'].tail(36)
    new_port_3yr_returns = returns_df['New Port Returns'].tail(36)
    corr_port = np.corrcoef(spy_3yr_returns.to_numpy(), port_3yr_returns.transpose().to_numpy())[1,0]
    corr_new_port = np.corrcoef(spy_3yr_returns.to_numpy(), new_port_3yr_returns.transpose().to_numpy())[1,0]
    stats_df.loc[30] = ['3 yr SPY Correlation', corr_port, corr_new_port, np.nan]

    #3 yr correlation with AGG
    agg_ticker = 'AGG US Equity' 
    agg_3yr_returns = returns[agg_ticker].tail(36)
    corr_port = np.corrcoef(agg_3yr_returns.to_numpy(), port_3yr_returns.transpose().to_numpy())[1,0]
    corr_new_port = np.corrcoef(agg_3yr_returns.to_numpy(), new_port_3yr_returns.transpose().to_numpy())[1,0]
    stats_df.loc[31] = ['3 yr AGG Correlation', corr_port, corr_new_port, np.nan]

    # Adding score column
    # Calculate Score for "CAGR"
    stats_df.loc[stats_df['stat'] == 'CAGR', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Complete Return"
    stats_df.loc[stats_df['stat'] == 'Complete Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])/2

    # Calculate Score for "Max Outperformance"
    stats_df.loc[stats_df['stat'] == 'Max Outperformance', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Max Drawdown"
    stats_df.loc[stats_df['stat'] == 'Max Drawdown', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Total Neg Return"
    stats_df.loc[stats_df['stat'] == 'Total Neg Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])/2

    # Calculate Score for "Total Pos Return"
    stats_df.loc[stats_df['stat'] == 'Total Pos Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])/2

    # Calculate Score for "Best Port Month"
    stats_df.loc[stats_df['stat'] == 'Best Port Month', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Worst Port Month"
    stats_df.loc[stats_df['stat'] == 'Worst Port Month', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Best Port Month"
    #stats_df.loc[stats_df['stat'] == 'Best New Port Month', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Worst New Port Month"
    #stats_df.loc[stats_df['stat'] == 'Worst New Port Month', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 1 yr Sharpe"
    stats_df.loc[stats_df['stat'] == 'Last 1 yr Sharpe', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 1 yr Sortino"
    stats_df.loc[stats_df['stat'] == 'Last 1 yr Sortino', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 1 yr Volatility"
    stats_df.loc[stats_df['stat'] == 'Last 1 yr Volatility', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 2 yr Sharpe"
    stats_df.loc[stats_df['stat'] == 'Last 2 yr Sharpe', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 2 yr Sortino"
    stats_df.loc[stats_df['stat'] == 'Last 2 yr Sortino', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 2 yr Volatility"
    stats_df.loc[stats_df['stat'] == 'Last 2 yr Volatility', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 3 yr Sharpe"
    stats_df.loc[stats_df['stat'] == 'Last 3 yr Sharpe', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 3 yr Sortino"
    stats_df.loc[stats_df['stat'] == 'Last 3 yr Sortino', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Last 3 yr Volatility"
    stats_df.loc[stats_df['stat'] == 'Last 3 yr Volatility', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Sharpe"
    stats_df.loc[stats_df['stat'] == 'Sharpe', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Sortino"
    stats_df.loc[stats_df['stat'] == 'Sortino', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Volatility"
    stats_df.loc[stats_df['stat'] == 'Volatility', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "2019 Return"
    stats_df.loc[stats_df['stat'] == '2019 Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "2020 Return"
    stats_df.loc[stats_df['stat'] == '2020 Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "2021 Return"
    stats_df.loc[stats_df['stat'] == '2021 Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "2022 Return"
    stats_df.loc[stats_df['stat'] == '2022 Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "2023 Return"
    stats_df.loc[stats_df['stat'] == '2023 Return', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Pos Return Months"
    stats_df.loc[stats_df['stat'] == 'Pos Return Months', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Calculate Score for "Neg Return Months"
    stats_df.loc[stats_df['stat'] == 'Neg Return Months', 'score'] = (stats_df['new_port'] - stats_df['old_port'])

    # Create a datetime object as a string (in desired format)
    # first_date_string = (datetime(2020, 3, 31, 0, 0, 0)).strftime("%Y-%m-%d %H:%M:%S")
    first_date_string = (datetime(2020, 3, 31, 0, 0, 0)).strftime("%Y-%m-%d")
    #print(first_date_string)

    second_date_string = (datetime(2022, 4, 29, 0, 0, 0)).strftime("%Y-%m-%d")
    #print(second_date_string)

    third_date_string = (datetime(2022, 9, 30, 0, 0, 0)).strftime("%Y-%m-%d")
    #print(third_date_string)

    fourth_date_string = (datetime(2023, 9, 30, 0, 0, 0)).strftime("%Y-%m-%d")
    #print(fourth_date_string)

    first_value_old_portfolio = returns_df['Port Returns'].loc[first_date_string]
    first_value_new_portfolio = returns_df['New Port Returns'].loc[first_date_string]
    #print(first_value)

    stats_df.loc[32] = ['COVID-19 Crash', first_value_old_portfolio, first_value_new_portfolio, np.nan, (first_value_new_portfolio - first_value_old_portfolio) * 100]

    second_value_old_portfolio = returns_df['Port Returns'].loc[second_date_string]
    second_value_new_portfolio = returns_df['New Port Returns'].loc[second_date_string]

    stats_df.loc[33] = ['Inflation Concerns', second_value_old_portfolio, second_value_new_portfolio, np.nan, (second_value_new_portfolio - second_value_old_portfolio) * 100]

    third_value_old_portfolio = returns_df['Port Returns'].loc[third_date_string]
    third_value_new_portfolio = returns_df['New Port Returns'].loc[third_date_string]

    stats_df.loc[34] = ['Interest Rate Hikes', third_value_old_portfolio, third_value_new_portfolio, np.nan, (third_value_new_portfolio - third_value_old_portfolio) * 100]

    fourth_value_old_portfolio = returns_df['Port Returns'].loc[third_date_string]
    fourth_value_new_portfolio = returns_df['New Port Returns'].loc[third_date_string]

    stats_df.loc[35] = ['Market Correction', fourth_value_old_portfolio, fourth_value_new_portfolio, np.nan, (fourth_value_new_portfolio - fourth_value_old_portfolio) * 100]
    stats_df.loc[36] = ['Sum', np.nan, np.nan, np.nan, stats_df['score'].sum()]

    return stats_df

@st.cache_data(show_spinner = False)
def write_results_to_excel_file(prices_df, returns_df, stats_df, client_id, asset_ticker):
    
    filename = 'quant_results/stats_' + str(client_id) + '_' + asset_ticker + '.xlsx'
    # excel_file_list.append(filename)

    with pd.ExcelWriter('quant_results/stats_' + str(client_id) + '_' + asset_ticker + '.xlsx') as writer:
        prices_df.to_excel(writer, sheet_name='Prices')
        returns_df.to_excel(writer, sheet_name='Returns')
        stats_df.to_excel(writer, sheet_name='Stats', index=False)

    return filename

@st.cache_data(show_spinner = False)
def get_unique_kristal_name(kristal_mappings):
    kristal_name = list(set(kristal_mappings['kristal_name']))

    return kristal_name

@st.cache_data(show_spinner = False)
def get_unique_client_ids(portfolios):
    client_ids = list(set(portfolios['client_id']))
    print(client_ids)
    
    return client_ids

@st.cache_data(show_spinner = False)
def show_particular_portfolio(portfolios, target_client_id):
    # Use boolean indexing to filter the rows with the specified client_id
    filtered_df = portfolios[portfolios['client_id'] == target_client_id]

    return filtered_df

@st.cache_data(show_spinner = False)
def download_data_as_excel_link(filepath):
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="results.xlsx">Download excel file</a>'
    st.markdown(link, unsafe_allow_html=True)
