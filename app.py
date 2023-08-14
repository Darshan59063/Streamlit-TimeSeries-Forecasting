# Importing Libraries and necessary tools
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import numpy
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def set_page(page: str) -> None:
    """
        Set the current page in the application session state

        This function updates the 'page' value in the session state with the provided page

        Parameters:
            page (str): The name of the page to be set in the session state

        Returns:
            None
    """
    st.session_state.page = page
    st.session_state["flag"] = 1


def page1():
    # print("page1 started")
    if "dataframe" not in st.session_state:
        st.session_state["dataframe"] = None
    if "date" not in st.session_state:
        st.session_state["date"] = None
    if "date_against" not in st.session_state:
        st.session_state["date_against"] = None
    if "flag" not in st.session_state:
        st.session_state["flag"] = None
    if "files" not in st.session_state:
        st.session_state["files"] = None
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None

    st.markdown("<h1 style='text-align: center'>Upload CSV files</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    files_names = list()
    files_names.append(None)

    if st.session_state["flag"]:
        for file in st.session_state["files"]:
            files_names.append(file.name)

        default_file_index = files_names.index(st.session_state["selected_file"])
        selected_file = st.selectbox("Select File", options=files_names, index=default_file_index)
        if selected_file is not st.session_state["selected_file"]:
            st.session_state["selected_file"] = selected_file
            st.session_state["date"] = None
            st.session_state["date_against"] = None

        if st.session_state["selected_file"]:

            file_obj = [file for file in st.session_state["files"] if file.name == st.session_state["selected_file"]][0]

            st.markdown(f"selected_file is {file_obj.name}")
            buffer = BytesIO(file_obj.getvalue())
            df = pd.read_csv(buffer)
            st.session_state["dataframe"] = df

            # Get the column names from the DataFrame
            column_names = df.columns.tolist()
            column_names.insert(0, None)

            default_file_index = column_names.index(st.session_state["date"])
            st.session_state["date"] = st.selectbox(f"\n\n From {file_obj.name} choose TimeStamp Column Name: ",
                                                    options=column_names, index=default_file_index)

            if st.session_state["date"]:

                column_names_excluding_timestamp = [col for col in column_names if col != st.session_state["date"]]
                default_file_index = column_names_excluding_timestamp.index(st.session_state["date_against"])
                st.session_state["date_against"] = st.radio("Select Entity Against Date",
                                                            options=column_names_excluding_timestamp, index=default_file_index)
                if st.session_state["date_against"]:
                    submit = st.button("Next", on_click=lambda: set_page("page2"))

    else:
        st.session_state["files"] = st.file_uploader("Upload Multiple CSV Files", type=['xlsx', 'csv'], accept_multiple_files=True)
        if st.session_state["files"]:
            # print(2)
            for file in st.session_state["files"]:
                files_names.append(file.name)
            # if st.session_state["selected_file"] is None:
#             #     print(3)
            st.session_state["selected_file"]  = st.selectbox("Select File", options=files_names)
            # st.session_state["selected_file"] = selected_file
            # else:
            #     default_file_index = files_names.index(st.session_state["selected_file"])
            #     st.session_state["selected_file"] = st.selectbox("Select File", options=files_names, index=default_file_index)
            if st.session_state["selected_file"]:
                # print(4)
                # Get the selected file object

                # default_file_index = files_names.index(st.session_state["selected_file"])
                # st.session_state["selected_file"] = st.selectbox("Select File", options=files_names,
                #                                                  index=default_file_index)
                file_obj = [file for file in st.session_state["files"] if file.name == st.session_state["selected_file"]][0]
                # print(type(file_obj))
                st.markdown(f"selected_file is {file_obj.name}")
                buffer = BytesIO(file_obj.getvalue())
                df = pd.read_csv(buffer)
                st.session_state["dataframe"] = df
                # print(st.session_state["dataframe"])

                # Get the column names from the DataFrame
                column_names = df.columns.tolist()
                column_names.insert(0, None)

#                 # print(1)
                # if "date" not in st.session_state:
                #     st.session_state["date"] = None
#                 # print(st.session_state["date"])

                # if st.session_state["date"] is None:
#                 #     print(5)
                st.session_state["date"] = st.selectbox(f"\n\n From {file_obj.name} choose TimeStamp Column Name: ",
                                                            options=column_names)
                # st.session_state["date"] = selected_file
                if st.session_state["date"]:
                    # print(6)
                    # default_file_index = column_names.index(st.session_state["date"])
                    # st.session_state["date"] = st.selectbox(f"\n\n From {file_obj.name} choose TimeStamp Column Name: ",
                    #                                         options=column_names, index=default_file_index)
                    # st.session_state["date"] = timestamp_column_name
                    # print(st.session_state["date"])
                    # Exclude a timestamp column
                    column_names_excluding_timestamp = [col for col in column_names if col != st.session_state["date"]]
                    # if "date_against" not in st.session_state:
                    #     st.session_state["date_against"] = None

                    # if st.session_state["date_against"] is None:
#                     #     print(7)
                    st.session_state["date_against"] = st.radio("Select Entity Against Date",
                                                        options=column_names_excluding_timestamp)
                    if st.session_state["date_against"]:
                        # print(8)
                        # default_file_index = column_names_excluding_timestamp.index(st.session_state["date_against"])
                        # st.session_state["date_against"] = st.radio("Select Entity Against Date",
                        #                                             options=column_names_excluding_timestamp, index=default_file_index)
                        submit = st.button("Next", on_click=lambda: set_page("page2"))
                        # print(submit)
                        # print(9)


def page2():
    # print(9)
    st.title("Data Visualization")
    if st.session_state['dataframe'] is not None and st.session_state['date'] is not None and st.session_state[
        'date_against'] is not None:
        # Read the file using pandas
        df = st.session_state['dataframe']
        print(df)
        df[st.session_state['date_against']].fillna('', inplace=True)
        # Convert the date column to datetime
        df[st.session_state['date']] = pd.to_datetime(df[st.session_state['date']], errors='coerce')
        df[st.session_state['date']] = df[st.session_state['date']].dt.strftime('%d/%m/%Y')
        df[st.session_state['date_against']] = pd.to_numeric(df[st.session_state['date_against']], errors='coerce')

        median_value = df[st.session_state['date_against']].median()
        df[st.session_state['date_against']].fillna(median_value, inplace=True)

        result = seasonal_decompose(df[st.session_state['date_against']], model='additive', period=12)

        trend_df = result.trend.to_frame(name='Trend')
        df = pd.concat([df, trend_df], axis=1)

        seasonality_df = result.seasonal.to_frame(name='Seasonality')
        df = pd.concat([df, seasonality_df], axis=1)

        residuals_df = result.resid.to_frame(name='Residual')
        df = pd.concat([df, residuals_df], axis=1)

        # Define custom colors for each line label
        line_colors = {
            f"{st.session_state['date_against']}": 'blue',
            'Trend': 'yellow',
            'Seasonality': 'red',
            'Residual': 'green'
        }
        # Create the time series plot using Plotly
        fig = px.line(df, x=st.session_state['date'],
                      y=[st.session_state['date_against'], 'Trend', 'Seasonality', 'Residual'])
        for line in fig['data']:
            label = line['name']
            line['line']['color'] = line_colors.get(label, 'black')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.title("Visualizing Trend, Seasonality and Residual")

        # Trend plot
        st.subheader("*Trend*")
        fig_trend = px.line(df, x=st.session_state['date'], y='Trend')
        fig_trend.update_traces(line=dict(color='yellow'))
        # fig_trend.update_layout(
        #     title='Trend Plot'
        # )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Seasonality plot
        st.subheader("*Seasonality*")
        fig_seasonality = px.line(df, x=st.session_state['date'], y='Seasonality')
        fig_seasonality.update_traces(line=dict(color='red'))
        # fig_seasonality.update_layout(
        #     title='Seasonality Plot'
        # )
        st.plotly_chart(fig_seasonality, use_container_width=True)

        # Residuals plot
        st.subheader("*Residuals*")
        fig_residuals = px.line(df, x=st.session_state['date'], y='Residual')
        fig_residuals.update_traces(line=dict(color='green'))
        # fig_residuals.update_layout(
        #     title='Residuals Plot'
        # )
        st.plotly_chart(fig_residuals, use_container_width=True)

        submit1 = st.button("Previous", on_click=lambda: set_page("page1"))
        submit2 = st.button("Next", on_click=lambda: set_page("page3"))


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
    diff.append(value)
    return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
def page3():
    # print(13)
    st.title("Forecast")
    df = st.session_state['dataframe']
    # x_labels = df[st.session_state['date']]
    # y_labels = df[st.session_state['date_against']]
    # x_labels = pd.to_datetime(st.session_state['date_against'], format='mixed')
    time_series = df[[st.session_state['date_against'],st.session_state['date']]]
    time_series.set_index(st.session_state['date'], inplace=True)
    print(time_series)
    # st.write(time_series)
    # Display original time series data
    st.subheader('Original Data')
    st.line_chart(time_series)

    p, d, q = 1, 1, 1  # Example values, adjust as needed
    arima_model = ARIMA(time_series, order=(p, d, q))
    results = arima_model.fit()
    print(results.summary())
    n = 10  # Number of steps to forecast
    forecast_freq = 'D'
    forecasted_values = results.forecast(steps=n)
    # Print the forecasted values
    st.write("Forecasted Values:", forecasted_values)
    print("Forecasted Values:", forecasted_values)

    # Create a new index for forecasted values
    forecast_index = pd.date_range(start=time_series.index[-1], periods=n + 1, freq=forecast_freq)[1:]



    # st.subheader('Original Data and Forecasted Values')
    # # forecast_index = forecast_index.tolist()
    # fig, ax = plt.subplots()
    #
    # # Convert datetime index to numerical representation for compatibility
    # date_num = date2num(forecast_index)
    # ax.plot_date(date_num, forecasted_values, color='red', label='Forecasted Values')
    # ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    #
    # plt.xlabel('Time')
    # plt.ylabel('Y Labels')
    # plt.title('Original Data and Forecasted Values')
    # plt.legend()

    # st.pyplot(fig)
    submit = st.button("Previous", on_click=lambda: set_page("page2"))


def navigation() -> None:
    """
        Function that manages the navigation between different pages in the application.

        If 'page' is not present in the session_state, it initializes 'page' with 'page1'.
        Based on the value of 'page' in the session_state, the corresponding page function is called.

        Returns:
            None
    """

    # If 'page' is not present in the session_state, it initializes 'page' with 'page1'
    if 'page' not in st.session_state:
        st.session_state['page'] = 'page1'

    # Based on the value of 'page' in the session_state, the corresponding page function is called
    if st.session_state['page'] == 'page1':
        page1()

    elif st.session_state['page'] == 'page2':
        page2()

    elif st.session_state['page'] == 'page3':
        page3()


# Code will execute only if the file was run directly and not imported
if __name__ == '__main__':
    # The control flow begins at this point
    print('start')

    # Set the page configuration for the Streamlit web application
    st.set_page_config(
        page_title="TimeSeries Forecasting",
        page_icon="📈"
    )

    navigation()
    print('-------------end----------------')
