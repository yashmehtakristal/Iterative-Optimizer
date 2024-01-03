# Importing various libraries
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import scale
from datetime import date
from dateutil.relativedelta import relativedelta
import io
import streamlit as st
import base64 
#from cryptography.fernet import Fernet

@st.cache_data(show_spinner = False)
def reading_returns():
    '''
    Reads adjusted_asset_returns.xlsx (named as returns) and does certain filtering by date
    
    Input: None
    Output: Returns dataframe
    '''

    # Read excel file
    returns = pd.read_excel('input_data/adjusted_asset_returns.xlsx')

    return returns

@st.cache_data(show_spinner = False)
def returns_manipulation(returns, start_date, end_date):

    # Convert to datetime and set Date column as index
    returns['Date']= pd.to_datetime(returns['Date'])
    returns.set_index('Date', inplace=True)

    # Start date and end date (currently hardcoded). Need to provide a date slider in streamlit UI
    # start_date = pd.to_datetime('2019-03-29')
    # end_date = pd.to_datetime('2023-08-31')

    # This will select only rows corresponding to the start and end date
    returns = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]

    # Sorts by column named "Date" in ascending order
    returns = returns.sort_values(by = ['Date'])

    # Boolean dataframe where element is True if not NaN
    # all(axis = 0) checks along each column and returns True if all values in a column are True
    # returns.loc[:,...] This selects all rows and only columns where corresponding element in Boolean series is True
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


#Every asset can be evaluated on 3 parameters - CAGR, Vol and Sharpe. The evaluation is done with 90% weight given to 
#old portfolio and 10% to the new asset. Creating such a portfolio and analysing it studies the correlations etc implicitly

@st.cache_data(show_spinner = False)
def evaluate_assets(segment_indexes, port_returns, port_sharpe, cagr, vol, returns):
    
    # Calculating length of segment_indexes list
    segment_num_indexes = len(segment_indexes)
    
    # Initializing list with zeors to store changes in statistics for each asset segment
    sharpe_change = [0]*segment_num_indexes
    cagr_change = [0]*segment_num_indexes
    vol_change = [0]*segment_num_indexes
    corr = [0]*segment_num_indexes
    
    # Calling the gen_drift_series function for portfolio returns
    portfolio_drift_series = gen_drift_series(port_returns[port_returns.columns[0]], 0.9)
    
    for i in range(segment_num_indexes):
        
        # Calling the gen_drift_series function for segment indexes
        index_drift_series = gen_drift_series(returns[segment_indexes[i]], 0.1)
        
        # Calculates new_port_prices by adding index_drift_series to an existing portfolio_drift_series
        # It's simulating how the portfolio's prices change when the returns of a specific segment are added to it.
        new_port_prices = np.add(portfolio_drift_series, index_drift_series)
        
        # This is the length of "new_port_prices"
        len_new_port_prices = len(new_port_prices)
        
        # This calculates the % change (returns) of the new_port_prices
        new_port_returns = pd.DataFrame(new_port_prices).pct_change()
        
        # Takes the last value in new_port_prices
        new_end_price = new_port_prices[len_new_port_prices-1]
        
        # Compounding the final price (final value of investment)/(starting value of investment, assumed to be 1)
        # over given number of compounding periods per year
        new_cagr = np.power(new_end_price,(12/(len_new_port_prices-1)))-1
        
        # Calculates annualized volatility by first calculating standard deviation of 'new_port_returns' and then annualizing it
        new_vol = np.std(new_port_returns, ddof=1)*np.sqrt(12)
        
        # Calculates sharpe by cagr/volatility
        new_port_sharpe = new_cagr/new_vol
        
        # Calculates change of sharpe, cagr and volatility with just portfolio
        sharpe_change[i] = (new_port_sharpe - port_sharpe)/port_sharpe
        cagr_change[i] = (new_cagr - cagr)/cagr
        vol_change[i] = (vol - new_vol)/vol
        
        # Calculates the correlation between returns of the current segment and overall portfolio returns
        corr[i] = np.corrcoef(returns[segment_indexes[i]].to_numpy(), port_returns.transpose().to_numpy())[1,0]
        
    # Creates various dataframes
    index_names = pd.DataFrame(segment_indexes)
    sharpe_change = pd.DataFrame(sharpe_change)
    cagr_change = pd.DataFrame(cagr_change)
    vol_change = pd.DataFrame(vol_change)
    corr = pd.DataFrame(corr)
    
    # Concatenates these above dataframes along columns to create index_stats dataframe
    index_stats = pd.concat([index_names, sharpe_change, cagr_change, vol_change, corr], join = 'outer', axis = 1)
    
    return index_stats


#This function is called for empty portfolios
@st.cache_data(show_spinner = False)
def evaluate_assets_only(segment_indexes, returns):
    
    # Calculating length of segment_indexes list
    segment_num_indexes = len(segment_indexes)
    
    # Initializing list with zeors to store changes in statistics for each asset segment
    sharpe_change = [0]*segment_num_indexes
    cagr_change = [0]*segment_num_indexes
    vol_change = [0]*segment_num_indexes
    corr = [0]*segment_num_indexes
    
    
    for i in range(segment_num_indexes):
        
        # Calling the gen_drift_series function for segment indexes
        index_drift_series = gen_drift_series(returns[segment_indexes[i]], 1)
        
        # Calculating length of index_drift_series in variable len_prices
        len_prices = len(index_drift_series)
        
        # This calculates the % change (returns) of the index_drift_series
        index_returns = pd.DataFrame(index_drift_series).pct_change()
        
        # Remove the last row (as it might possibly have NaN values) and resetting index
        index_returns = index_returns.tail(-1).reset_index()
        
        # Selects all rows and all columns starting from the second column 
        index_returns = index_returns.iloc[: , 1:]
         
        # Retrieves last value (ending price) in index_drift_series
        end_price = index_drift_series[len_prices-1]
        
        # Calculates change of sharpe, cagr and volatility with just portfolio
        cagr_change[i] = np.power(end_price,(12/(len_prices-1)))-1
        vol_change[i] = np.std(index_returns, ddof=1)*np.sqrt(12)
        sharpe_change[i] = cagr_change[i]/vol_change[i]
        
    # Creates various dataframes
    index_names = pd.DataFrame(segment_indexes)
    sharpe_change = pd.DataFrame(sharpe_change)
    cagr_change = pd.DataFrame(cagr_change)
    vol_change = pd.DataFrame(vol_change)
    corr = pd.DataFrame(corr)
    
    # Concatenates these above dataframes along columns to create index_stats dataframe
    index_stats = pd.concat([index_names, sharpe_change, cagr_change, vol_change, corr], join = 'outer', axis = 1)
    
    return index_stats

@st.cache_data(show_spinner = False)
def gen_clients_ranking(portfolios, client_id, kristal_mappings, asset_universe, returns):
    
    # Only return the particular rows where client_id column has particular client_id value and reset the index
    portfolio1 = portfolios.loc[portfolios['client_id'] == client_id].reset_index()
        
    # Select all rows
    # as well as all columns except the first column (starting from second column onwards)
    portfolio1 = portfolio1.iloc[: , 1:]
    
    # Getting mappings data for the portfolio assets
    # merges portfolio1 dataframe with kristal_mappings on kristal_Id
    # Left join - All rows from portfolio1 are retained, but matching rows in kristal_mappings are added.
    portfolio1 = portfolio1.merge(kristal_mappings, on = 'kristal_id', how = 'left')

    # Calculates a list of 'kristal_ticker' values that are present in 'portfolio1' but not in 'asset_universe'.
    assets_not_found = list(set(portfolio1['kristal_ticker']) - set(asset_universe))

    
    # Error handling for the above
    if (len(assets_not_found) != 0):
        print("Assets not found")
        print(assets_not_found)
        
    # 'portfolio1' is filtered to keep only rows where kristal_ticker is in asset_universe 
    # and then index is resetted
    portfolio1 = portfolio1[portfolio1['kristal_ticker'].isin(asset_universe)].reset_index()

    # Calculate total portfolio value by summing up values in 'AUM in USD' column and storing it in portfolio_value
    portfolio_value = portfolio1['AUM in USD'].sum()
    
    # Calculate the weight of each asset in the portfolio
    portfolio1['weight'] = portfolio1['AUM in USD']/portfolio_value
    

    # Creating drifting Portfolio prices based on $1 starting value
    
    # Obtaining number of assets in portfolio by looking at the number of rows in portfolio1
    num_assets = portfolio1.shape[0]
    
    # Length of 'returns' dataframe
    num_returns = len(returns)
    
    # Initializing list of port_prices
    port_prices = [0]*(num_returns+1)
    
    # For loop iterating num_assets times
    for i in range(num_assets):
        #print(portfolio1['asset'][i])
        
        # Call function gen_drift_series with returns of particular portfolio and their weights
        # Then, add the series of price changes generated by gen_drift_series
        port_prices = np.add(port_prices, gen_drift_series(returns[portfolio1['kristal_ticker'][i]], portfolio1['weight'][i]))
    
    
    # Calculating the length of the portfolio prices
    len_prices = len(port_prices)
    print(len_prices)
    
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

    # Creating portfolio_stats dataframe with 'metric' and 'value' columns 
    portfolio_stats = pd.DataFrame()
    portfolio_stats['metric'] = ['Sharpe', 'CAGR', 'Vol']
    portfolio_stats['value'] = [float(port_sharpe.iloc[0]), cagr, float(vol.iloc[0])]

    # Call the evaluate_assets function 
    multiple_scores = evaluate_assets(asset_universe, port_returns, port_sharpe, cagr, vol, returns)
    print(len(multiple_scores))
    
    # If portfolio1 is empty, call evaluate_assets_only() function
    if portfolio1.empty:
        multiple_scores = evaluate_assets_only(returns = returns, segment_indexes = asset_universe)
    
    # Renaming columns
    multiple_scores.columns = ['kristal_ticker', 'sharpe_score', 'cagr_score', 'vol_score', 'correlation']
    
    # Dropping NA values in particular columns
    multiple_scores = multiple_scores.dropna()

    # Calculating combined score and scaled combined score according to pre-determined formula
    multiple_scores['combined_score'] = 0.5*multiple_scores['sharpe_score'] + 0.4*multiple_scores['vol_score'] + 0.1*multiple_scores['cagr_score']
    multiple_scores['scaled_combined_score'] = scale(multiple_scores['combined_score'])

    # Getting mappings data for the portfolio assets (kristal_mappings)
    # merges multiple_scores dataframe with kristal_mappings on kristal_id
    # Left join - All rows from kristal_mappings are retained, but matching rows in kristal_mappings are added.
    multiple_scores = kristal_mappings.merge(multiple_scores, on='kristal_ticker', how='left')

    # Filtering out bonds that are about to expire (1 month from now)
    today = date.today()
    one_month_later = today + relativedelta(months=1)
    bonds_about_to_expire = kristal_mappings.loc[(kristal_mappings['asset_maturity'] < pd.to_datetime(one_month_later, format='%Y-%m-%d'))]
    
    # Boolean Series that checks whether each 'kristal_ticker' value in multiple_scores is present in the set of 'kristal_ticker' values from bonds_about_to_expire
    # The "~" operator then checks if kristal_ticker is NOT in this set (i.e. selects False values - those not in kristal_ticker)
    multiple_scores = multiple_scores.loc[~multiple_scores['kristal_ticker'].isin(set(bonds_about_to_expire['kristal_ticker']))]
    
    # Filter out rows in multiple_scores where the 'kristal_sub_type' is 'STOCK' and reset index
    multiple_scores = multiple_scores.loc[multiple_scores['kristal_sub_type'] != 'STOCK'].reset_index(drop=True)

    # num_scores is the length of multiple_scores
    num_scores = len(multiple_scores)
    
    # Initialize a list of per_match_col
    per_match_col = [0]*num_scores
    
    #scores_list = list(multiple_scores['sharpe_vol_score'])
    
    # Take list of scaled_combined_score
    scores_list = list(multiple_scores['scaled_combined_score'])
    
    # Take maximum and minimum of scores_list
    overall_max_score = max(scores_list)
    overall_min_score = min(scores_list)
    
    # Include only elements that are less than or equal to 1 
    # or greater than or equal to -1
    scores_list[:] = [x for x in scores_list if x <= 1]
    scores_list[:] = [x for x in scores_list if x >= -1]
    
    # Find the new maximum and minimum of scores_list
    max_score = max(scores_list)
    min_score = min(scores_list)

    # Finds the difference between overall_max
    # and max (with condition) and multiplies the difference by 0.05
    # Takes either the calculated value or a maximum value of 0.05
    upper_limit_gap = min(((overall_max_score-max_score)*0.05), 0.05)
    print(upper_limit_gap)
    
    # Upper_limit is the difference between 0.99 and upper_limit_gap
    upper_limit = 0.99-upper_limit_gap
    
    # iterating from 0 to num_scores
    for i in range(num_scores):
        
        # Retrieves scaled_combined_score at corresponding index i
        score = multiple_scores['scaled_combined_score'][i]
        
        # If score greater than 1
        if score > 1:
            
            # Calculate per_match value
            per_match = ((score - max_score)/(overall_max_score - max_score)) * upper_limit_gap + upper_limit
            
            if per_match > 0.99:                
                per_match = 0.99
        
        # Else if, score is less than -1
        elif score < -1:
            
            # Calculate per_match value
            per_match = (score+3)/2 * 0.05
            
            if per_match < 0:
                per_match = 0 
        
        # Else if
        # Score > -1
        # and, score < 1
        else:
            
            # Use particular formula of per_match based on certain conditions
            if overall_max_score == max_score:
                
                if overall_min_score == min_score:
                    per_match = ((score-min_score)/(max_score-min_score))*upper_limit
                else: 
                    per_match = ((score-min_score)/(max_score-min_score))*(upper_limit-0.05) + 0.05
            
            else:
                per_match = ((score-min_score)/(max_score-min_score))*(upper_limit-0.05)+0.05
        
        # Set it 0.59 if below condition matches
        if ((multiple_scores['sharpe_score'][i]<0) and (multiple_scores['cagr_score'][i]<0) and (multiple_scores['vol_score'][i]<0) and (per_match>0.59)):
            per_match = 0.59
        
        # Add it to corresponding index number
        per_match_col[i] = per_match 

    # Assign the series to multiple_scores column
    multiple_scores['per_match'] = pd.Series(per_match_col)
    
    # Sort multiple_scores with highest combined score (descending order)
    multiple_scores = multiple_scores.sort_values(by = ['combined_score'], ascending=False)

    filepath = 'quant_results/results_' + str(client_id) + '.xlsx'

    # excel_file_list.append(filename)

    # Write it to excel file with client_id in end
    with pd.ExcelWriter('quant_results/results_' + str(client_id) + '.xlsx') as writer:
        
        # Write portfolio1 dataframe to 'Portfolio' sheet
        portfolio1.to_excel(writer, sheet_name='Portfolio')
        
        # Write portfolio1 dataframe to 'Stats' sheet
        portfolio_stats.to_excel(writer, sheet_name='Stats')
        
        # Write portfolio1 dataframe to 'Scores' sheet
        multiple_scores.to_excel(writer, sheet_name='Scores')
    
    return portfolio1, portfolio_stats, multiple_scores, filepath



@st.cache_data(show_spinner = False)
def read_portfolios():
    '''
    Read excel file (client_data_saurabh.xlsx) as portfolios

    Input: None
    Output: None
    '''
    
    #Reading input file - client portfolios
    portfolios = pd.read_excel('input_data/client_data.xlsx', 'Portfolio')

    return portfolios

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
def read_kristal_mappings(returns):
    '''
    Reads Asset Index mapping excel file as kristal_mappings after applying certain conditions

    Input: None
    Output: kristal_mappings dataframe, asset_universe, adjusted_returns_universe
    '''

    # Reading input file - Asset Index mapping 
    kristal_mappings = pd.read_excel('input_data/Asset Index mapping.xlsx')

    # notna() checks for each row in 'kristalid' column if value is not null or missing
    # .loc[] selects only rows where 'kristalid' column is not null
    # Resets the row indices to default integer indices
    kristal_mappings = kristal_mappings.loc[kristal_mappings['kristalid'].notna()].reset_index()

    # Selecting only particular columns and renaming them
    kristal_mappings = kristal_mappings[['kristalid', 'kristal_name', 'kristal_sub_type', 'Asset', 'Index', 'Maturity Asset']]
    kristal_mappings.columns = ['kristal_id', 'kristal_name', 'kristal_sub_type', 'kristal_ticker', 'kristal_index', 'asset_maturity']

    # Converting asset maturity to datetime format
    kristal_mappings['asset_maturity'] = pd.to_datetime(kristal_mappings['asset_maturity'], format='%d/%m/%y')

    # Extracting unique values from kristal_ticker (Asset) column
    asset_universe = list(set(kristal_mappings['kristal_ticker']))

    # Get the column headers (first row) of the returns dataframe file
    adjusted_returns_universe = list(set(returns.columns.values))

    # Define unique list which now contains only the unique asset identifiers that are present in both the 'kristal_ticker'
    # column of kristal_mappings and the column names of the returns DataFrame.
    asset_universe = list(set(asset_universe) & set(adjusted_returns_universe))

    return kristal_mappings, asset_universe, adjusted_returns_universe

@st.cache_data(show_spinner = False)
def download_data_as_excel_link(portfolio1, portfolio_stats, multiple_scores, filepath):
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="results.xlsx">Download excel file</a>'
    st.markdown(link, unsafe_allow_html=True)

@st.cache_data(show_spinner = False)
def download_data_as_excel_link_old(portfolio1, portfolio_stats, multiple_scores, filepath):

    # buffer to use for excel writer
    buffer = io.BytesIO()

    # writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

    # Create an Excel writer
    writer = pd.ExcelWriter(buffer, engine='xlsxwriter')

    # Taking a dataframe and converting to excel
    portfolio1.to_excel(writer, index=False, header=True, sheet_name = 'Portfolio')
    portfolio_stats.to_excel(writer, index=False, header=True, sheet_name='Stats')
    multiple_scores.to_excel(writer, index=False, header=True, sheet_name='Scores')

    workbook = writer.book
    writer.close()
    processed_data = buffer.getvalue()

    # Save the Excel writer
    # writer.save()

    # reset pointer
    # buffer.seek(0)

    # b64 = base64.b64encode(buffer.read()).decode()

    link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{processed_data}" download="results.xlsx">Download excel file</a>'

    st.markdown(link, unsafe_allow_html=True)

