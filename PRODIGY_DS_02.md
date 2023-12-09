# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Mount Google Drive for dataset access in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Load dataset into a Pandas DataFrame
df = pd.read_csv('/content/drive/MyDrive/UCI_Credit_Card.csv')

# Display dataset information
print(df.info())

# Display summary statistics of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Define categorical variables for analysis
categorical_vars = ['SEX', 'EDUCATION', 'MARRIAGE']

# Specify colors for each category in visualizations
colors = ['skyblue', 'lightcoral']

# Visualize distribution of categorical variables with respect to the target variable
for var in categorical_vars:
    sns.countplot(x=var, hue='default.payment.next.month', data=df, palette=colors)
    plt.title(f'Distribution of {var} and Default Status')
    plt.show()

# Define numerical variables for analysis
numerical_vars = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']

# Visualize distribution of numerical variables using histograms
for var in numerical_vars:
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')
    plt.show()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Visualize the distribution of the target variable
sns.countplot(x='default.payment.next.month', data=df, color='skyblue')
plt.title('Distribution of Default Status')
plt.show()


# Print a message indicating successful execution of the code
print("Analysis and visualizations completed successfully.")
