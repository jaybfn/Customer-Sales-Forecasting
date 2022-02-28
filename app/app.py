# Importing Libraries

# libraries for webapp
import streamlit as st
#libraries for EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplcyberpunk
import plotly.figure_factory as ff
import plotly.io as pio
import datetime

# Machine Learning and Deep Learning Libraries
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import LSTM ,Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from keras.regularizers import L1L2
from tensorflow.keras.models import load_model
from numpy.random import seed


## Data exploration 
#st.set_page_config(layout="wide")

# creates a sidebar on the webapp with multiple subpage options
option = st.sidebar.selectbox("Choose the following options",("Data Analysis", "Sales Forecasting","Custom Forecasting"))

# condition to enter the first page
if option == "Data Analysis":
    # title of the first page
    st.title('Customer Retail Sales Forecasting')
    # created a space between the main title and the rest of the page
    st.write("---")
    """
    ### Data Source:
    
    """
    st.markdown("""[TravelSleek:]('https://travelsleek.in/') It's a family-owned E-commerce business that specializes 
    in personalized travel products which are manufactured from Faux Leather.""")

    # creates a dropdown box 
    with st.expander("Products"):
     st.write("""
         Customized Products.
     """)
     images = ['../plots/item1.jpg','../plots/item2.jpg', '../plots/item3.jpg']
     st.image(images, width = 200, use_column_width=False) #caption=["some generic text"] * len(images)
     st.markdown("""[ImageSource:](https://travelsleek.in/)""")
    
    
    st.write('_____________________________')
    """
    ### Tech Stack
    """
    # creating columns in the webapp so that the text or images can be arranged in the column format
    col1, col2, col3 = st.columns(3)
    with col1:
        """
        ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

        ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

        ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

        ![Visual Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)

        """
    with col2:
        """
        ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

        ![Seaborn](https://img.shields.io/badge/Seaborn-.-orange?style=for-the-badge)

        ![Matplotlib](https://img.shields.io/badge/Matplotlib-.-red?style=for-the-badge)

        ![mplcyberpunk](https://img.shields.io/badge/mplcyberpunk-.-yellow?style=for-the-badge)

        """

    with col3:
        """
        ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

        ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
        
        ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

        ![Streamlit](https://img.shields.io/badge/Streamlit-.-brightgreen?style=for-the-badge)
        """
    
##########################################################################################


################### From plotly express ###################################################
    st.write('_____________________________')
    st.header(option)
    """
    ### Info:
    - Data was exported from E-commerce platform to .csv file.
    - Timeline of the data availability is from October 2019 to Jan 2022.
        - October 2019 to October 2020 on Instagram Market Place.
        - November 2020 to Jan 2022 on E-commerce platform.
    - The size of the data was 8000 sales data points and 77 features associated with it.
    - Nearly, 36% of the data was filled with null val!!!!
    - Total SKU in the Inventory : 35 SKU

    """
    st.write('_____________________________')
    st.subheader('Top5 selling products and their Revenue Generation')

    # calling the .csv file
    Top5 = pd.read_csv('../forecast_data/Top5.csv')
    Top5 = Top5.sort_values(by = 'count', ascending=False)
    Top5_rev = pd.read_csv('../forecast_data/Top5_rev.csv')
    Top5_rev = Top5_rev.sort_values(by = 'Tot', ascending=False)
    # barplot 
    fig2 = px.bar(Top5, x='product', 
                y='count',text_auto='.3s',
                title="Top 5 Selling Products",
                labels = {'product':'Products',
                        'count':'Quantity'})
    fig2.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
    fig2.update_layout( xaxis = dict( tickfont = dict(size=18)),font_size = 16)
    fig2.update_layout( yaxis = dict( tickfont = dict(size=18)))
    st.plotly_chart(fig2)
    st.write('_ _ _ _ _ _ _ ')

    fig1 = px.bar(Top5_rev, x='product', 
                y='Tot',text_auto='.4s',
                title="Revenue Generated from Top 5 Selling Products",
                labels = {'product':'Products',
                        'Tot':'Revenue (INR)'})
    fig1.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
    fig1.update_layout( xaxis = dict( tickfont = dict(size=18)), font_size = 16)
    fig1.update_layout( yaxis = dict( tickfont = dict(size=18)))
    st.plotly_chart(fig1)
    st.write('_____________________________')
    pie = pd.read_csv('../forecast_data/pie.csv')
    st.subheader('Overall Revenue Contribution form Top 5 Selling Products')
    
    fig_pie = px.pie(pie, values='Percentage_Rev', 
                    names='product',
                    color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie)
    """
    - Total SKU in the Inventory : 35 SKU
    """
    
    st.subheader('Around 30% of the total sales is generated by Top 5 selling product')
    st.write('_____________________________')
    st.subheader('Where are my Customers from?')
    states = pd.read_csv('../forecast_data/states.csv')
    #st.map(states)
    # scatter plot
    fig_map = px.scatter_geo(states, lat="latitude", 
                            lon = "longitude",
                            hover_name="States",
                            size="Counts", 
                            projection="natural earth",
                            color = 'States', width = 800, height = 600 
                     ) #
    fig_map.update_geos(#projection_type="orthographic",
                        resolution=50, lataxis_showgrid=True, lonaxis_showgrid=True,
                        showcoastlines=True, coastlinecolor="RebeccaPurple",
                        showland=True, landcolor="White",
                        showocean=True, oceancolor="LightBlue",scope="world",
                        showcountries=True, countrycolor="grey",
                        showsubunits=True, subunitcolor="Blue",
                        fitbounds="locations"
                        ) #showlakes=True, lakecolor="LightBlue",
                        #showrivers=True, rivercolor="LightBlue"
    fig_map.update_layout(mapbox_style="open-street-map",height=400 ,margin={"r":0.5,"t":0.5,"l":0,"b":0})
    st.plotly_chart(fig_map)

    st.write('_____________________________')

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
# page 2
if option == "Sales Forecasting":
    st.title("Customer Sales Forecasting Using Bidirectional LSTM model")

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:

        st.subheader('A simple architecture of LSTM cell:')
        st.image('../plots/LSTM.jpeg',width = 400, use_column_width=True)
        st.markdown("""[Image Source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fhackernoon%2Funderstanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4&psig=AOvVaw3U-5Qi_LKvjOLt6ipTjkbz&ust=1646083003957000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCPj5mKvnoPYCFQAAAAAdAAAAABAJ)""")
    
        """
        ### Trials:
        
        """

    show_code = st.checkbox('Show LSTM Models')

    if show_code:
        
        code = """

Trial1: (activation function : relu, leaky-relu, tanh)
        model = Sequential()
        model.add(LSTM(64, activation = 'relu', input_shape = ( X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(y_train.shape[1]))

        Result:
        RMSE = 40000 

Trial2:
        model = Sequential()
        model.add(LSTM(256, activation = 'relu', input_shape = ( X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(128, activation = 'relu', return_sequences = True))
        model.add(LSTM(64,activation = 'relu', return_sequences = True))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dense(25, activation = 'relu'))
        model.add(Dense(y_train.shape[1]))
        
        Result:
        RMSE = 32000 

Trial3:
        Layer of Regularization 
        - L1
        - L2 
        - L1,L2
        
        Result:
        Underfitting result with a single value prediction.

Trial4: Best Model

        ___________________________________________________
         2 layers of Bi-LSTM, activation_function = 'tanh',
         1 dropout layer of 0.5 and two dense layer.
        ___________________________________________________

        model = Sequential()
        model.add(Bidirectional(LSTM(64, activation = 'tanh', 
                                input_shape = ( X_train.shape[1], X_train.shape[2]), 
                                return_sequences=True)))
        model.add(Bidirectional(LSTM(32, activation = 'tanh', 
                                return_sequences = False)))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation = 'tanh'))
        model.add(Dense(y_train.shape[1]))

        history = mod.fit(X_train,y_train, 
                    epochs = 150, 
                    batch_size = 8, 
                    validation_split=0.2, 
                    verbose = 1,
                    callbacks=[cb],
                    shuffle= True)

        Result:
        RMSE : 4600 

        """
        
        st.code(code, language='python')
        st.image('../plots/loss.png',width = 400, use_column_width=False)

    with col2:
        st.write('#')
        st.write('#')
        st.write('#')
        st.write('A single LSTM cell consists of 3 gates:')
        """
        - Forget Gate
        - Input Gate
        - Output Gate
        """
    
    forecast = pd.read_csv('../forecast_data/forecast_avg.csv')
    ########################################################################
    res = pd.read_csv('../data/results_week.csv', index_col=0)
    st.subheader("**Weekly Sales Data and Prediction**")
    #st.dataframe(data=res)
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots()
    ax.plot(res.week[:109], res.sales[:109] ,linewidth=1, markersize=5, label = 'Train Data')
    ax.plot(res.week[:109], res.sales_pred[:109], linewidth=1, markersize=5, label = 'Train Prediction' )
    ax.scatter(res.week[:109], res.sales[:109], alpha = 0.3)
    ax.legend()
  
    ax.plot(res.week[109:122], res.sales[109:122],linewidth=1, markersize=5 ,label = 'Test Data')
    ax.plot(res.week[109:122], res.sales_pred[109:122],linewidth=1, markersize=5 ,label = 'Test Prediction' )
    ax.scatter(res.week[109:122], res.sales[109:122],alpha = 0.3)
    ax.legend()

    ax.plot(forecast['week'], forecast['mean'], '-',label = 'Forecasting')
    ax.fill_between(forecast['week'], forecast['mean'] - forecast['std']*2, forecast['mean'] + forecast['std']*2, alpha=0.2)
    ax.plot(forecast['week'], forecast['mean'], '*')
    ax.legend()

    plt.xlabel('weeks')
    plt.ylabel('Sales (INR)')

    t = ax.text(80, 20000, "Training_Data", ha="center", va="center", rotation=0, size=10,
    bbox=dict(boxstyle="square,pad=0.3",fc="black" ,ec="w", lw=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("square", pad=0.3)

    t = ax.text(115, 20000, "Testing_Data", ha="center", va="center", rotation=90, size=10,
    bbox=dict(boxstyle="square,pad=0.3",fc="black" ,ec="w", lw=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("square", pad=0.3)

    t = ax.text(127, 20000, "Forecasting", ha="center", va="center", rotation=90, size=10,
    bbox=dict(boxstyle="square,pad=0.3",fc="black" ,ec="w", lw=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("square", pad=0.3)

    mplcyberpunk.add_glow_effects()
    st.pyplot(fig)

###########################################################################
 
###########################################################################

###########################################################################
    forecast = pd.read_csv('../forecast_data/forecast_avg.csv')

    #st.markdown("**Here comes some random data**")
    ####st.dataframe(data=forecast)
    
    st.write("---")
    #st.header("Some plotting")
    #st.subheader("Plotting with matplotlib and seaborn")
    #forecast = pd.read_csv('../data/forecast_12weeks.csv')
    fig, ax = plt.subplots()
    
    ax.plot(res.week[105:122], res.sales[105:122],linewidth=1, markersize=5 ,label = 'Test Data')
    ax.plot(res.week[105:122], res.sales_pred[105:122],linewidth=1, markersize=5 ,label = 'Test Prediction' )
    ax.scatter(res.week[105:122], res.sales[105:122],alpha = 0.3)
    ax.legend()

    t = ax.text(113, 45000, "Testing_Data" , ha="center", va="center", rotation=0, size=8,
    bbox=dict(boxstyle="square,pad=0.3",fc="black" ,ec="w", lw=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("square", pad=0.3)

    ax.plot(forecast['week'], forecast['mean'], '-', label = 'Forecasting')
    ax.fill_between(forecast['week'], forecast['mean'] - forecast['std']*2, forecast['mean'] + forecast['std']*2, alpha=0.2) #95% CI
    ax.plot(forecast['week'], forecast['mean'], 'o')
    ax.legend()
    t = ax.text(127.5, 45000, "forecasting for 12 weeks (95% CI)" , ha="center", va="center", rotation=0, size=8,
    bbox=dict(boxstyle="square,pad=0.3",fc="black" ,ec="w", lw=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("square", pad=0.3)

    plt.xlabel('weeks')
    plt.ylabel('Sales (INR)')
    mplcyberpunk.add_glow_effects()
    st.pyplot(fig)
########################################################################


    # st.subheader("Plotting with plotly-chart")
    # fig =px.line(
    #     data_frame=res,
    #     x='week',
    #     y='sales',
    #     markers = True)
    # st.plotly_chart(fig)

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

if option == "Custom Forecasting":
    res = pd.read_csv('../data/results_week.csv', index_col=0)
    
    # moving average calculation
    def moving_average_(dataframe, window_size):
        
        #window_size = 4
        #tot = sales['Total].tolist()
        if window_size != 1:
            i = 0
            moving_averages = []
            while i < len(dataframe['Total']) - window_size + 1:
                this_window = dataframe['Total'][i : i + window_size]
                window_average = sum(this_window) / window_size
                moving_averages.append(window_average)
                i += 1

            sales = pd.DataFrame(moving_averages, columns=['Total'])

        else: 
            moving_averages = dataframe['Total']
            sales = pd.DataFrame(moving_averages, columns=['Total'])
        return sales

    #adding lag to the sales for multistep forecasting
    def lags(dataframe, lags):
        for lag in range(1,lags):
            col_name = 'lag_' +str(lag)
            dataframe[col_name] = dataframe['sales_norm'].shift(lag)
        #drop null val
        dataframe = dataframe.dropna().reset_index(drop = True)
        return dataframe

    def forecasting(dataframe, window_size, lag, n_future):
        
        #dataframe = sales_
        #window_size = 4
        #lags = 6
        #n_future = number of weeks /days to forecast in future

        model = load_model(r'..\models\model_final.h5') 
        scaler = MinMaxScaler()
        for n in range(n_future):
        
            sales = pd.DataFrame(moving_average_(dataframe, window_size), columns=['Total'])
            sales_norm = scaler.fit_transform(sales.Total.values.reshape(-1, 1))
            sales_norm = sales_norm.flatten().tolist()
            sales['sales_norm'] = sales_norm
            sales = sales.drop(['Total'],axis = 1)
            sales = lags(sales,lag)
            last_row = sales[-1:]
            last_row = last_row.drop(['sales_norm'], axis = 1)
            last_row = last_row.to_numpy()
            last_row = last_row.reshape(last_row.shape[0], 1, last_row.shape[1])
            pred = model.predict(last_row)
            forecast = scaler.inverse_transform(pred)[:,0]#.tolist()
            forecast = pd.DataFrame(forecast, columns = ['Total'])
            dataframe = pd.concat([dataframe, forecast], ignore_index=True)#
            subset = pd.DataFrame(dataframe.tail(n_future))
        # fig_forecast = px.scatter(subset, x = np.arange(n_future),y = 'Total')
        # st.plotly_chart(fig_forecast)
        return subset

    sales_tot = pd.read_csv('../data/sales_full.csv', index_col=0)
    #sales_tot.head()
    sales_tot = sales_tot.drop(['year','week','weeks'], axis = 1)
    sales_ = sales_tot.copy()

 
    st.subheader('Play with the parameters to get weekly forecast: ')
    st.write('_____________________________')
    Moving_Avg = st.slider('Moving Average:', 1, 10)
    st.write('Moving_Avg', Moving_Avg)
    st.write('#')
    n_futures = st.slider('Number of Weeks to forecast:', 1, 20)
    st.write('weeks:', n_futures)

    with st.container():
        

        if st.button('Forecast'):
            data = forecasting(sales_,Moving_Avg, 6, n_futures)

            test_forecast = res['sales_pred'].tail(12).to_list() + data['Total'].to_list()
            #st.write(test_forecast)

            test = pd.DataFrame(test_forecast, columns=['sales'])
            #new = pd.concat([test, data], axis=0)
            #st.write(test)
            pio.templates.default = "simple_white"
            fig_forecast = px.line(test, x = np.arange(len(test)),
                                    y = test['sales'],
                                    markers = True,
                                    labels = {'x':'weeks'}) #,color=test['sales']

            fig_forecast.add_shape(type="line", 
                                        line_color="salmon", 
                                        line_width=3, 
                                        opacity=1, 
                                        line_dash="dot",
                                        x0=11, x1=11,  
                                        y0=test['sales'].min(), 
                                        y1=test['sales'].max())

            fig_forecast.add_annotation( # add a text callout with arrow
            text="Test Data", x=5, y=45000)
            fig_forecast.add_annotation( # add a text callout with arrow
            text="Forecasting", x=17, y=62500)
            fig_forecast.update_layout( xaxis = dict( tickfont = dict(size=18)),font_size = 18)
            fig_forecast.update_layout( yaxis = dict( tickfont = dict(size=18)))
            st.plotly_chart(fig_forecast)

