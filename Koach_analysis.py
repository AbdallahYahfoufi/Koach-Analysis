import streamlit as st
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu 
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
from streamlit_card import card

from PIL import Image

import json
import base64
import requests
import random

from streamlit_lottie import st_lottie
#from streamlit_toggle import st_toggle_switch
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.colored_header import colored_header
from streamlit_extras.function_explorer import function_explorer
from streamlit_extras.metric_cards import style_metric_cards
#data exploration
import pandas as pd
import numpy as np
import calendar
import requests
from bs4 import BeautifulSoup as bs
import re
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#

urll="https://drive.google.com/file/d/1Ak9dfbqpVNu6F1TWS4iWVojbIegOWRyy/view?usp=sharing"
koachlogo='https://drive.google.com/uc?export=download&id='+urll.split('/')[-2]
st.set_page_config(layout="wide",page_title="Data-Driven Analytics",page_icon= koachlogo)

def load_data(path):
    df=pd.read_csv(path)
    return df

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
#-----------------------------------------------------------------------------#
# Menu Configuration
with st.sidebar:
    st.markdown("""
    <style>
        .side-by-side {
            display: flex;
            justify-content: space-between;
        }
    </style>
    <div class="side-by-side">
        <div style="text-align:center;">
            <img src="https://drive.google.com/uc?id=1qcwhWj62I_t7qApTgIzsbZ_ftLsOHq05" height="150" alt="im1">
        </div>
        <div style="text-align:center;">
            <img src="https://drive.google.com/uc?export=download&id=1Ak9dfbqpVNu6F1TWS4iWVojbIegOWRyy" height="150" alt="im2">
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    tabs = option_menu('',['Introduction','Data Analytics', 'Market Basket Analysis', 'Forecasting Analytics'], 
    icons=['caret-right-square-fill','clipboard-data', 'shop','graph-up-arrow'], 
    default_index=0,styles={"container": {"padding": "0!important", "background-color": "#FFFFFF"},
        "icon": {"color": "black", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"3px", "--hover-color": "#FFFFFF"},
        "nav-link-selected": {"background-color": "#FF4B4B"},})
#----------------------------------------------------------------------------#
# Component1: Home
if tabs == "Introduction":    
 col1, col2 = st.columns([3,2])
 with col1:
    title = "<h1 style='text-align:center;color:#FF4B4B; font-size:25px;'>Koach Outlet: Dressing for Success, One Data-Driven Decision at a Time</h1>" 
    st.markdown(title, unsafe_allow_html=True)
    html="""<p style="font-family: Arial;font-size:21px;line-height:1.8em;color: #333333;"> Koach Outlet is a clothing store that offers a wide range of products at discounted prices. 
    The company operates out of a brick & mortar store in Beirut and provides delivery services all over Lebanon, thus
possessing a large customer base, and it is important for Koach Outlet to understand its customers'
purchasing patterns in order to improve its profitability.</p>"""
    st.markdown(html, unsafe_allow_html=True)
    html2 ="""<p style="font-family: Arial;font-size:21px;line-height:1.8em;color: #333333;">The app aims to provide a decision-making tool for tracking sales and customer purchase behavior. It will feature:
<li style="font-family: Arial;font-size:21px;line-height:1.8em;color: #333333;"><b>Sales Dashboard:</b> Showcasing buying behaviors at the product and year level and offering a comparative study of inventory management.</li>
<li style="font-family: Arial;font-size:21px;line-height:1.8em;color: #333333;"><b>Item Recommendations:</b> An approach to recommend items based on customer transactions</li>
<li style="font-family: Arial;font-size:21px;line-height:1.8em;color: #333333;"><b>Sales Forecasting:</b> Predicting sales interactively
</li></p>"""
    st.markdown(html2,unsafe_allow_html=True)

    #col1, col2 = st.columns([3,1])
 with col2:
        file_ = open("shop_sales.jpeg", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
           f'<img src="data:image;base64,{data_url}" alt="img" width="620" height= "690">',
         unsafe_allow_html=True,) 
 st.write("")
 st.write("")
 st.write("")
 hide_streamlit_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
 st.markdown(hide_streamlit_style, unsafe_allow_html=True)

 show_streamlit_style = """
        <style>
        footer:after {
            content: content;
            visibility: visible;
        }
        </style>
        <div class="content">
        <h6 style="font-size: 14px;color: black;white-space:nowrap;"><b>Made with&nbsp;<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="14">&nbsp;by Abdallah Yahfoufi</b><h6>
        <h6 style="font-size: 14px;color: black;white-space:nowrap;"><b>Let's connect&nbsp;</b><a href="https://www.linkedin.com/in/abdallahyahfoufi/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn logo" height="14"></a></h6>
        
        </div>
        """
 st.markdown(show_streamlit_style, unsafe_allow_html=True)       
#########################################################################################
# Component 2: Data Analytics
if tabs == "Data Analytics":
    #read dataframes and data prep 
    Sales_Daily=load_data("Koach_Sales.csv")
    Sales_Daily['Date'] = pd.to_datetime(Sales_Daily['Date'], format='%d/%m/%Y', errors='coerce')
    Sales_Daily['Year'] = Sales_Daily['Date'].dt.year
    Sales_Daily['Year'] = Sales_Daily['Year'].astype('object')
    items_df = load_data("inv_management.csv")
    items_df['Date'] =  pd.to_datetime(items_df['Date'])
    items_df['Purchase Date'] =  pd.to_datetime(items_df['Purchase Date'])
    items_df['Inventory_Age'] = items_df['Date'] - items_df['Purchase Date']
    tab1, tab2 = st.tabs(["Sales Dashboard", "Inventory management"])
    with tab1:
      col1, col2, col3 = st.columns(3)
      min_value = 10
      max_value = 50
      with col1:
        year = st.selectbox("Year",options=(Sales_Daily.dropna().Year.unique()))
      with col3:
        top_items = st.number_input("Select number of top items to display:", min_value, max_value, 10)
      df_year = Sales_Daily[Sales_Daily['Year'] == year]
      total_sales = (Sales_Daily[Sales_Daily['Year'] == year]['Price in USD']).sum()
      #col11,col12,col13 = st.columns([1,1,1])
      with col2:
         sales = round (total_sales,2)
         #lottie_sales = load_lottieurl("https://lottie.host/cd23f818-308f-49a8-8abc-07c871b1cf44/loL4UeN58H.json")
         #st_lottie(lottie_sales, height=80)
         st.markdown(
        f"<h3 style='text-align: center;'>Total Sales</h3>",
        unsafe_allow_html=True,
    )
         st.markdown(
        f"<h3 style='text-align: center; font-weight: semi-bold;'>${sales}</h3>",
        unsafe_allow_html=True,
    )
         st.write("")
         st.write("")
      #############################################################################################################
      col1, col2, col3 = st.columns([2, 1, 2])
      with col1:
           data = df_year[df_year['Price in USD'] > 0]
           data['Month'] = data['Date'].dt.month_name()  # Extract month name
           monthly_sales = data.groupby(['Year', 'Month'])['Price in USD'].sum().reset_index().sort_values(by='Month', ascending=False)
           # Highlight top month
           top_month = monthly_sales[monthly_sales['Price in USD'] == monthly_sales['Price in USD'].max()]
           # Create the bar chart
           fig = px.bar(monthly_sales, x='Month', y='Price in USD',
             labels={'Price in USD': 'Sales in USD'}) 
           fig.update_layout(legend=dict(title=dict(text='')),
        xaxis=dict(categoryorder='array', categoryarray=['January', 'February', 'March','April', 'May', 'June', 'July',
                                                        'August', 'September', 'October', 'November', 'December']), 
        yaxis=dict(showgrid=False),
        title={'text':f'Monthly Sales for {int(year)}',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20} # Set the font size of the title
        },
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
    )
           fig.add_annotation(
    x=top_month['Month'].values[0],  # X-coordinate of the annotation
    y=top_month['Price in USD'].values[0],  # Y-coordinate of the annotation
    text=f"Top month is {top_month['Month'].values[0]} : {top_month['Price in USD'].values[0]:,.0f}",  # Text of the annotation
    showarrow=True,  # Show arrow
    arrowhead=1,  # Arrowhead style
    ax=0,  # X-coordinate of arrow tail
    ay=-25,  # Y-coordinate of arrow tail
    arrowwidth=3,  # Width of the arrow
    arrowcolor='red',  # Color of the arrow
)
           st.plotly_chart(fig, use_container_width=True)


      with col3:
      # Extract brand names
     # Create a scatter plot
        monthly_sales_cat = data.groupby(['Year', 'Month', 'Category'])[['Price in USD', 'Sales Qty']].sum().reset_index().sort_values(by=['Year', 'Month'], ascending=[False, False])
        fig21 = px.scatter(monthly_sales_cat, x='Month', y='Price in USD',labels={'Price in USD': 'Sales in USD'}, color='Category', size='Sales Qty')  # Adjust the opacity for better visibility)
        # Customize layout
        fig21.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=['January', 'February', 'March','April', 'May', 'June', 'July',
                                                        'August', 'September', 'October', 'November', 'December']), 
         title={'text':f'Sales Trend over {int(year)}',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20} # Set the font size of the title
        },
    xaxis_title='Month',
    yaxis_title='Total Sales',
    legend_title='Category',
    showlegend=True,
    paper_bgcolor='white',  # Set plot background color to white
    plot_bgcolor='white',  # Set plot area background color to white
)
     # Show the plot
        st.plotly_chart(fig21,use_container_width=True)
 
      col11, col12, col13 = st.columns([2, 1, 2])     
      with col11:       
      # Filter data for positive sales quantities
        filtered_sales = df_year[df_year['Sales Qty'] > 0]
    # Group data by Year and Description, summing the Sales Qty
        top_items_grouped = filtered_sales.groupby(['Year', 'Description'])['Sales Qty'].sum().reset_index()
    # Get top items for each year
        top_per_year = top_items_grouped.groupby('Year').apply(lambda x: x.nlargest(top_items, 'Sales Qty')).reset_index(drop=True)
    # Identify the most selling item for each year
        most_selling_item = top_per_year.groupby('Year').apply(lambda x: x.nlargest(1, 'Sales Qty')).reset_index(drop=True)
        # Use Plotly Express for the bar chart
        fig2 = px.bar(
    top_per_year,
    x='Sales Qty',
    y='Description',
    color='Description',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    labels={'Sales Qty': 'Total Sales Quantity', 'Description': 'Item Description'},
   
) 
   # Customize layout
        fig2.update_layout( 
        yaxis=dict(showgrid=False),
        title={'text':f'Top {top_items} Selling Items Over {year}',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20} # Set the font size of the title
        },
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        showlegend=False
    )
    # Show the plot
        st.plotly_chart(fig2, use_container_width=True)
      with col13:
       filtered_sales=df_year[df_year['Sales Qty'] > 0]    
       brand_counts = filtered_sales['Description'].str.split().str[-1].value_counts()
       filtered_sales['Description'] = filtered_sales['Description'].str.upper()
      # Create a DataFrame from the value counts
       brand_counts_df = pd.DataFrame({'Brand': brand_counts.index, 'Count': brand_counts.values})
       brand_counts_df ['Brand'] = brand_counts_df['Brand'].str.replace('UNDER ARMOUR', 'UNDERARMOUR')
       brand_counts_df ['Brand'] = brand_counts_df['Brand'].str.replace('NEW BALANCE', 'NEWBALANCE')
       brand_counts_df ['Brand'] = brand_counts_df['Brand'].str.replace('NB', 'NEWBALANCE')
       brand_counts_df = brand_counts_df[brand_counts_df['Brand'].str.isalpha()]
       # Create a Pie chart using Plotly Express
       fig11 = px.pie(brand_counts_df.head(top_items), names='Brand', values='Count')
       fig11.update_layout(legend=dict(title=dict(text='')), 
        yaxis=dict(showgrid=False),
        title={'text':f'Top {top_items} Brands for {int(year)}',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20} # Set the font size of the title
        },
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
    )
       st.plotly_chart(fig11,use_container_width=True)

    with tab2:
      text = """
<h1 style="font-size: 24px;color:#c90515;text-align: center">
    <div style="display: inline-block">
        Koach Inventory Analysis
    </div>
</h1>
"""
      st.markdown(text,unsafe_allow_html=True)
      col1, col3, col2 = st.columns(([4, 1, 4]))
      with col1:
        restocked_items=items_df[items_df['Date'] < items_df['Purchase Date']]
        restocked_items = restocked_items[restocked_items['year']==2023]
        restocked_items['Item Description']=restocked_items['Item Description'].str.split(n=1).str[1]
        fig1=px.bar(x=restocked_items['Item Description'].value_counts().values[:10], y=restocked_items['Item Description'].value_counts().index[:10],
             labels={'x': 'Qty', 'y': 'Items'},orientation='h')
        fig1.update_layout(
    title=f'Which items was restocked the most during 2023?', xaxis = dict(
        showgrid=False, 
    ), 
    yaxis = dict( 
        showgrid=False
    ))
        col1.plotly_chart(fig1,use_container_width=True)
        stocked_items = items_df[items_df['Date'] > items_df['Purchase Date']]
        stocked_items['Inventory_Age'] = stocked_items['Inventory_Age'].dt.days.astype("int")
        stocked_items = stocked_items[stocked_items['Price in USD']>0]
        fig2= px.scatter(stocked_items, x='Inventory_Age', y='Price in USD',labels={'Price in USD': 'Sales in USD'}, trendline='ols')
        fig2.update_layout(
    title=f'How does inventory age affect sale?',
    xaxis = dict(
        title='Inventory Age (in days)',
        showgrid=False, 
    ), 
    yaxis = dict(
        title='Sales', 
        showgrid=False
    ))
        col2.plotly_chart(fig2,use_container_width=True)
    #with st.expander("Show correlation",expanded=True):

      col1.write("**Restocked item scenarios** occur when, after a period of being out of stock, an item is replenished, and subsequent purchases are made from the replenished stock. We noticed that Reebok pants, Calvin Klein t-shirt kids and Under Armour sweatshirt are the most restocked items and hence, Koach should focus on always having such items in stock or purchase it from suppliers in advance"
)
      col2.write(
    "In the scatter plot, we observe a **relatively flat trend line**, indicating no significant correlation between the duration of items being in stock and their sale prices. It suggests that the duration of an item in stock doesn't strongly influence its selling price. Furthermore, we note that the majority of data points fall within the price range of less than 75 USD and a maximum duration of 200 days.")
      col2.write("**One plausible recommendation** for the company is to consider creating bundles at discounted prices. This strategy can be beneficial for customers, as they can enjoy cost savings when purchasing bundled items. Additionally, the company can explore selling these bundles directly at a higher price, leveraging the perceived value of acquiring multiple items together")
   
if tabs == "Market Basket Analysis":
            #load ARM data (check orders analysis notebook for more details)
            associations = load_data("Cust_basket.csv")
            associations = associations[['Customer Basket', 'Recommended Product']]
            #formatting customer basket column
            # Function to extract values
            def extract_values(row):
                basket = row['Customer Basket'].replace("frozenset({'", "").replace("'})", "").strip()
                product = row['Recommended Product'].replace("frozenset({'", "").replace("'})", "").strip()
                return basket, product
                # Apply the function to each row
            associations[['Customer Basket', 'Recommended Product']] = associations.apply(extract_values, axis=1, result_type='expand')
            st.markdown("""<p style="font-family:Georgia;"> In the following section, we will introduce the concept of <b>frequently bought together concept</b> where once you filter by a specific basket, you will get recommended item(s) accordingly.
    <br>Noting that the following recommendation system is robust and works best by having an online website to track customer transactions in real-time.</br></p>""", unsafe_allow_html=True)
            # Add custom CSS
            #with st.sidebar:
                #st.file_uploader("Upload your custom dataset", type=["csv"])
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align:center;color:#18D5D2;font-family:Georgia;font-size:25px;'>Filter By Basket</h3>", unsafe_allow_html=True)
            #basket = st.selectbox("Select a basket",(associations['Customer basket'].unique()))
            # Randomly select a basket from the available customer baskets
            random_basket = random.choice(associations['Customer Basket'])
            recommended_item = associations[associations['Customer Basket'] == random_basket].iloc[0]['Recommended Product']
            if st.button("Recommend"):
                  st.write(associations[associations['Customer Basket'] == random_basket]['Customer Basket'].head(1))
                  st.markdown("<h3 style='text-align:center;color:#18D5D2;font-family:Georgia;font-size:25px;'>You may also like:</h3>", unsafe_allow_html=True)
                  st.write(recommended_item)



if tabs == "Forecasting Analytics":
    expander = st.expander("Click me if you want to know more🙋🏻‍♂️")
    expander.markdown("""<b>Interpreting the Forecast Plot:</b>
    <ul>
        <li>The blue line represents the predicted values of the time series.</li>
        <li>The shaded blue area around the line represents the uncertainty around the predictions. The width of the shaded area represents the size of the confidence interval.</li>
        <li>The black dots represent the actual values of the time series, if available.</li>
        <li>If the predicted values closely match the actual values, the model is likely a good fit for the data. If there are significant discrepancies between the predicted and actual values, the model may not be a good fit.</li>
    </ul>""", unsafe_allow_html=True)

    text = """
<h1 style="font-size: 24px;color:#c90515;text-align: center">
    <div style="display: inline-block">
        Filters 
        <i class="far fa-question-circle" title="{txt}"></i>
    </div>
</h1>
"""
    txt = "Interact to forecast sales per category for next month or 6 months and tune the model with paramaters implemented here"
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)
    st.sidebar.markdown(text.format(txt=txt), unsafe_allow_html=True)
    # Preprocess data
    Sales_Daily=load_data("Koach_Sales.csv")
    Sales_Daily['Date'] = pd.to_datetime(Sales_Daily['Date'], format='%d/%m/%Y', errors='coerce')
    Sales_Daily = Sales_Daily[Sales_Daily['Price in USD'] >= 0]
    ts_df = Sales_Daily[['Date', 'Category', 'Price in USD']]
    ts_df = ts_df.groupby(['Category', pd.Grouper(key='Date', freq='M')]).sum().reset_index()
    ts_df.columns = ['category', 'ds', 'y']
    #Get category input from user
    category = st.sidebar.selectbox('Select Category:', ts_df['category'].unique())
    #get your season mode
    mode = st.sidebar.selectbox('Seasonality mode:',['multiplicative','additive'])
    # Create and fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode=mode)
    # Filter by selected category
    df_category = ts_df[ts_df['category'] == category]
    # Get forecast horizon input from user
    forecast_horizon = st.sidebar.selectbox('Select forecast horizon (in months):', [1,6,12])
    # Get user input for hyperparameters
#     #explained here for reference
#     Seasonality Prior Scale:

# Seasonality refers to the recurring patterns or cycles in a time series, often associated with certain time intervals like daily, weekly, or yearly patterns.
# The "seasonality prior scale" parameter in Prophet controls the strength of the seasonality components in the model. A higher seasonality prior scale makes the model more flexible and allows it to fit the training data more closely, potentially capturing more intricate patterns. However, it may also lead to overfitting, especially if the training data contains noise.
# On the other hand, a lower seasonality prior scale makes the model more conservative, smoothing out the seasonal patterns and making the model less sensitive to short-term fluctuations.
# Changepoint Prior Scale:

# Changepoints are points in the time series where the data exhibits a significant change in its trend. Prophet models changepoints as potential locations where the time series behavior changes.
# The "changepoint prior scale" parameter influences the flexibility of the model in detecting and incorporating changepoints. A higher changepoint prior scale makes the model more sensitive to potential changepoints, allowing it to adapt more quickly to changes in the data. However, this can also lead to overfitting if there are too many changepoints, capturing noise in the data.
# A lower changepoint prior scale makes the model less responsive to changes and results in smoother trend predictions, reducing the risk of overfitting but potentially missing some real changepoints.
    seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale', min_value=0.001, max_value=10.0, value=0.05, step=0.001)
    # Set hyperparameters 
    model.seasonality_prior_scale = seasonality_prior_scale
    model.changepoint_prior_scale = changepoint_prior_scale
    # Fit model to category data
    model.fit(df_category)
    # Generate future dates
    future = model.make_future_dataframe(periods=forecast_horizon, freq='M')
    # Generate forecasts
    forecast = model.predict(future)
    # Filter predictions for future dates only
    forecast_future = forecast[['ds', 'yhat']].tail(forecast_horizon)
    # Plot actual vs predicted values for past and future dates
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(title=f"Actual vs Predicted Sales ({category} - Past and Future Dates)")
    st.plotly_chart(fig1)
    # Plot Prophet components for past and future dates
    fig2 = plot_components_plotly(model, forecast)
    fig2.update_layout(title=f"Prophet Components ({category} - Past and Future Dates)")
    st.plotly_chart(fig2)
