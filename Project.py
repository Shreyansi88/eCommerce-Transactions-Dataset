import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import io

# Function to load CSV from URL
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

# Display basic information about each dataset
print("Products Dataset:")
print(products.info())
print("\nCustomers Dataset:")
print(customers.info())
print("\nTransactions Dataset:")
print(transactions.info())

# Summary statistics
print("\nProducts Summary:")
print(products.describe())
print("\nCustomers Summary:")
print(customers.describe())
print("\nTransactions Summary:")
print(transactions.describe())

# Check for missing values
print("\nMissing Values:")
print(products.isnull().sum())
print(customers.isnull().sum())
print(transactions.isnull().sum())

# Convert data types
products['Price'] = products['Price'].astype(float)
transactions['Quantity'] = transactions['Quantity'].astype(int)
transactions['TotalValue'] = transactions['TotalValue'].astype(float)
transactions['Price'] = transactions['Price'].astype(float)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])

# Visualize the distribution of customers by region
plt.figure(figsize=(10, 6))
customers['Region'].value_counts().plot(kind='bar')
plt.title('Distribution of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()

# Analyze product categories
plt.figure(figsize=(12, 6))
products['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Products by Category')
plt.axis('equal')
plt.show()

# Analyze transaction values over time
transactions.set_index('TransactionDate', inplace=True)
monthly_sales = transactions.resample('M')['TotalValue'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Merge datasets for further analysis
merged_data = transactions.reset_index().merge(customers, on='CustomerID').merge(products, on='ProductID')

# Analyze top selling products
top_products = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(descending=True).head(10)
plt.figure(figsize=(12, 6))
top_products.plot(kind='bar')
plt.title('Top 10 Selling Products')
plt.xlabel('Product Name')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze customer purchasing behavior
customer_purchase_freq = merged_data.groupby('CustomerID').size()
plt.figure(figsize=(10, 6))
sns.histplot(customer_purchase_freq, kde=True)
plt.title('Distribution of Customer Purchase Frequency')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.show()

print("EDA completed. Please check the generated visualizations.")