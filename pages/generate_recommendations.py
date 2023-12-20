# All imports

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Iterative Optimizer", page_icon = "🧠", layout = "wide")
st.header("🧠 Iterative Optimizer")

add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

import openai
import pandas as pd

## Importing functions
from core.part1_assets_ranking import get_unique_client_ids, read_portfolios, show_particular_portfolio, returns_manipulation, reading_returns, read_portfolios, read_kristal_mappings, gen_clients_ranking, download_data_as_excel_link

### CODE

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY

# Error handling for OpenAI API key
if not openai_api_key:
    st.warning(
        "There is something wrong with the API Key Configuration."
        "Please check with creator of the program (OpenAI keys can be found at https://platform.openai.com/account/api-keys)"
    )


if 'username' in st.session_state:
    st.session_state.username = st.session_state.username

# st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# Display app only if user is logged in
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Call the get_unique_client_ids function
    portfolios = read_portfolios()
    client_ids = get_unique_client_ids(portfolios)

    # Allow user to select client id
    client_id_selected = st.selectbox(label = "Please select the client id from dropdown", options = client_ids, index = None, key = "select_client_id", help = "select_client_id", placeholder="Choose particular client id", disabled = False, label_visibility = "visible")

    if client_id_selected:

        # Select filtered_df
        filtered_df = show_particular_portfolio(portfolios, target_client_id = client_id_selected)

        # Display dataframe containing final results
        st.dataframe(data = filtered_df, use_container_width = True, column_order = None)

    # Allow user to select date range
    start_date = pd.to_datetime('2019-03-29')
    end_date = pd.to_datetime('2023-08-31')

    date_selected = st.date_input(
        label = "Please select a date range for which you want to analyze portfolio data for",
        value = (start_date, end_date),
        min_value = None,
        max_value = None,
        key = "select_date_input",
        help = "Please make sure to select a date range/interval",
        on_change = None,
        disabled = False,
        format = "DD/MM/YYYY",
        label_visibility = "visible"
        )
    
    if not isinstance(date_selected, tuple):
        print("Error: date_selected is not a tuple.")
        st.warning('date_selected is not a tuple', icon="⚠️")
        st.stop()

    else:
        if len(date_selected) != 2:

            if len(date_selected) == 0:
                print("Error: date_selected is an empty tuple.")
                st.warning('Please make sure to select a date range (and not leave it empty)', icon="⚠️")
                st.stop()

            else:
                print("Error: date_selected is not a tuple of length 2.")
                st.warning('Please make sure to select an interval of dates (that is, both start date and end date)', icon="⚠️")
                st.stop()

        # Length of tuple is equal to 2
        else:

            # Convert the tuple of dates to the desired format
            selected_start_date = pd.to_datetime(date_selected[0]).strftime('%Y-%m-%d')
            selected_end_date = pd.to_datetime(date_selected[1]).strftime('%Y-%m-%d')

    # If user clicks on the button process
    if st.button("Run", type = "primary"):

        if not client_id_selected:
            st.warning("Please select client id", icon = "⚠️")
            st.stop()

        if client_id_selected:

            # with st.spinner("Generating Recommendations"):
            #     returns = reading_returns()
            #     returns = returns_manipulation(returns, start_date, end_date)
            #     portfolio = read_portfolios()
            #     kristal_mappings, asset_universe, adjusted_returns_universe = read_kristal_mappings()
            #     excel_file_list = []
            #     num_clients = len(client_ids)
            #     for i in range(num_clients):
            #         gen_clients_ranking(client_ids[i])

            with st.spinner("Reading asset returns excel file"):
                returns = reading_returns()
                returns = returns_manipulation(returns, start_date, end_date)

            st.success("Successfully read asset returns excel file", icon="✅")

            with st.spinner("Reading portfolios excel file"):
                portfolio = read_portfolios()

            st.success("Successfully read portfolios excel file", icon="✅")

            with st.spinner("Reading Asset Index mapping excel file"):
                kristal_mappings, asset_universe, adjusted_returns_universe = read_kristal_mappings(returns)

            st.success("Successfully read Asset Index mapping excel file", icon="✅")

            with st.spinner("Generating Recommendations:"):
                
                # Store excel file list
                excel_file_list = []

                #num_clients = len(client_ids)
                    
                # Calling gen_clients_ranking function on particular client_ids 
                portfolio1, portfolio_stats, multiple_scores, filepath = gen_clients_ranking(portfolios, client_id = client_id_selected, kristal_mappings = kristal_mappings, asset_universe = asset_universe, returns = returns)

            st.success("Successfully generated recommendations", icon="✅")

        # Display "Scores" excel sheet from results excel file
        st.dataframe(data = multiple_scores, use_container_width = True, column_order = None)

        # Download excel data
        download_data_as_excel_link(portfolio1, portfolio_stats, multiple_scores, filepath)

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
