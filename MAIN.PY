import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to load CSV from URL
@st.cache_data
def load_csv_from_url(url):
    response = urlopen(url)
    return pd.read_csv(io.StringIO(response.read().decode('utf-8')))

# Load the datasets
products_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Products-xfJGM6zskW2UT80n0SWOuwwwPfZAbN.csv"
customers_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Customers-WiUWrumewDjiqmRMfXPdmKKNQh4CgH.csv"
transactions_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Transactions-ygPtgqmkAyKm8KBuzx8qNgg0qPosp4.csv"

products = load_csv_from_url(products_url)
customers = load_csv_from_url(customers_url)
transactions = load_csv_from_url(transactions_url)

# Convert data types
products['Price'] = products['Price'].astype(float)
transactions['Quantity'] = transactions['Quantity'].astype(int)
transactions['TotalValue'] = transactions['TotalValue'].astype(float)
transactions['Price'] = transactions['Price'].astype(float)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])

# Merge datasets
merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Streamlit app
st.title('eCommerce Data Analysis and Prediction')

# Sidebar for navigation
page = st.sidebar.selectbox('Select a page', ['Data Overview', 'Exploratory Data Analysis', 'Predictive Model'])

if page == 'Data Overview':
    st.header('Data Overview')
    
    st.subheader('Products Dataset')
    st.write(products.head())
    st.write(f"Total products: {len(products)}")
    
    st.subheader('Customers Dataset')
    st.write(customers.head())
    st.write(f"Total customers: {len(customers)}")
    
    st.subheader('Transactions Dataset')
    st.write(transactions.head())
    st.write(f"Total transactions: {len(transactions)}")

elif page == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')
    
    st.subheader('Distribution of Customers by Region')
    fig, ax = plt.subplots(figsize=(10, 6))
    customers['Region'].value_counts().plot(kind='bar', ax=ax)
    plt.title('Distribution of Customers by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Customers')
    st.pyplot(fig)
    
    st.subheader('Distribution of Products by Category')
    fig, ax = plt.subplots(figsize=(10, 6))
    products['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    plt.title('Distribution of Products by Category')
    plt.axis('equal')
    st.pyplot(fig)
    
    st.subheader('Monthly Sales Over Time')
    monthly_sales = transactions.set_index('TransactionDate').resample('M')['TotalValue'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_sales.plot(ax=ax)
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    st.pyplot(fig)
    
    st.subheader('Top 10 Selling Products')
    top_products = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(descending=True).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_products.plot(kind='bar', ax=ax)
    plt.title('Top 10 Selling Products')
    plt.xlabel('Product Name')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader('Distribution of Customer Purchase Frequency')
    customer_purchase_freq = merged_data.groupby('CustomerID').size()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(customer_purchase_freq, kde=True, ax=ax)
    plt.title('Distribution of Customer Purchase Frequency')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Number of Customers')
    st.pyplot(fig)

elif page == 'Predictive Model':
    st.header('Predictive Model')
    
    # Prepare features and target
    X = merged_data[['Quantity', 'Price']]
    y = merged_data['TotalValue']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared Score: {r2:.2f}")
    
    st.subheader('Predict Total Value')
    quantity = st.number_input('Enter Quantity', min_value=1, value=1)
    price = st.number_input('Enter Price', min_value=0.01, value=100.00, step=0.01)
    
    if st.button('Predict'):
        future_transaction = pd.DataFrame({'Quantity': [quantity], 'Price': [price]})
        predicted_total_value = model.predict(future_transaction)
        st.write(f"Predicted total value: ${predicted_total_value[0]:.2f}")

st.sidebar.markdown('---')
st.sidebar.write('Created by Your Name')