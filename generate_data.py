import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

def generate_telecom_customer_data(num_customers=10000):
    """
    Generate realistic telecom customer data for ML training
    """
    np.random.seed(42)  # For reproducible results
    
    data = []
    
    for i in range(num_customers):
        # Customer Demographics
        customer_id = f"CUST_{i+1:06d}"
        age = np.random.normal(35, 12)
        age = max(18, min(80, int(age)))  # Keep age realistic
        
        # Account Information
        account_length = np.random.exponential(24)  # months, exponential distribution
        account_length = max(1, min(120, int(account_length)))
        
        # Usage Patterns
        monthly_minutes = np.random.lognormal(6, 1)  # Log-normal for realistic usage
        monthly_minutes = max(0, int(monthly_minutes))
        
        monthly_data_gb = np.random.exponential(5)  # GB per month
        monthly_data_gb = round(max(0.1, monthly_data_gb), 2)
        
        monthly_sms = np.random.poisson(50)  # SMS count follows Poisson
        
        # Financial Data
        monthly_charge = 20 + (monthly_minutes * 0.05) + (monthly_data_gb * 2) + (monthly_sms * 0.1)
        monthly_charge = round(monthly_charge, 2)
        
        # Payment behavior
        late_payments = np.random.poisson(0.5)  # Average 0.5 late payments
        if account_length < 6:
            late_payments = np.random.poisson(0.2)  # New customers pay better
        
        # Service usage patterns
        international_calls = np.random.exponential(2)
        international_calls = max(0, int(international_calls))
        
        customer_service_calls = np.random.poisson(1)  # Support calls per month
        
        # Contract type
        contract_type = np.random.choice(['Prepaid', 'Postpaid'], p=[0.6, 0.4])
        
        # Device information
        device_age_months = np.random.exponential(18)
        device_age_months = max(1, min(60, int(device_age_months)))
        
        # Generate churn indicator (target variable)
        # Higher probability of churn based on risk factors
        churn_probability = 0.05  # Base churn rate
        
        # Risk factors that increase churn probability
        if late_payments > 2:
            churn_probability += 0.3
        if customer_service_calls > 3:
            churn_probability += 0.2
        if monthly_charge > 80:
            churn_probability += 0.15
        if account_length < 3:
            churn_probability += 0.25
        if age < 25 or age > 65:
            churn_probability += 0.1
            
        churned = np.random.binomial(1, min(churn_probability, 0.8))
        
        # Generate fraud indicators
        fraud_probability = 0.02  # Base fraud rate
        if international_calls > 10:
            fraud_probability += 0.1
        if monthly_charge > 200:
            fraud_probability += 0.15
        if account_length < 1:
            fraud_probability += 0.2
            
        is_fraud = np.random.binomial(1, min(fraud_probability, 0.3))
        
        # Location data (for ATGhana context)
        ghana_cities = ['Accra', 'Kumasi', 'Tamale', 'Sekondi-Takoradi', 'Cape Coast', 
                       'Tema', 'Obuasi', 'Koforidua', 'Sunyani', 'Ho']
        city = np.random.choice(ghana_cities, p=[0.3, 0.2, 0.1, 0.08, 0.08, 0.07, 0.05, 0.04, 0.04, 0.04])
        
        customer_data = {
            'customer_id': customer_id,
            'age': age,
            'city': city,
            'account_length_months': account_length,
            'contract_type': contract_type,
            'monthly_minutes': monthly_minutes,
            'monthly_data_gb': monthly_data_gb,
            'monthly_sms': monthly_sms,
            'monthly_charge': monthly_charge,
            'late_payments': late_payments,
            'international_calls': international_calls,
            'customer_service_calls': customer_service_calls,
            'device_age_months': device_age_months,
            'churned': churned,
            'is_fraud': is_fraud
        }
        
        data.append(customer_data)
    
    return pd.DataFrame(data)

def generate_transaction_data(customer_df, transactions_per_customer=50):
    """
    Generate transaction data for fraud detection
    """
    transactions = []
    
    for _, customer in customer_df.iterrows():
        num_transactions = np.random.poisson(transactions_per_customer)
        
        for _ in range(num_transactions):
            # Transaction amount based on customer profile
            if customer['is_fraud']:
                # Fraudulent transactions tend to be higher and more varied
                amount = np.random.lognormal(4, 1.5)  # Higher amounts
            else:
                amount = np.random.lognormal(2, 0.8)  # Normal amounts
            
            amount = max(1, round(amount, 2))
            
            # Transaction time (random within last 3 months)
            days_ago = np.random.randint(0, 90)
            transaction_time = datetime.now() - timedelta(days=days_ago)
            
            # Transaction type
            transaction_types = ['Voice_Call', 'Data_Usage', 'SMS', 'International', 'Premium_Service']
            if customer['is_fraud']:
                # Fraudsters use more premium services and international
                transaction_type = np.random.choice(transaction_types, 
                                                  p=[0.2, 0.2, 0.1, 0.3, 0.2])
            else:
                transaction_type = np.random.choice(transaction_types, 
                                                  p=[0.4, 0.3, 0.2, 0.05, 0.05])
            
            transaction = {
                'customer_id': customer['customer_id'],
                'transaction_id': f"TXN_{len(transactions)+1:08d}",
                'amount': amount,
                'transaction_type': transaction_type,
                'timestamp': transaction_time,
                'is_fraud': customer['is_fraud']
            }
            
            transactions.append(transaction)
    
    return pd.DataFrame(transactions)

# Generate the datasets
print("Generating customer data...")
customer_data = generate_telecom_customer_data(10000)

print("Generating transaction data...")
transaction_data = generate_transaction_data(customer_data, 30)

# Save to CSV files
customer_data.to_csv('atghana_customers.csv', index=False)
transaction_data.to_csv('atghana_transactions.csv', index=False)

print(f"Generated {len(customer_data)} customer records")
print(f"Generated {len(transaction_data)} transaction records")
print(f"Churn rate: {customer_data['churned'].mean():.2%}")
print(f"Fraud rate: {customer_data['is_fraud'].mean():.2%}")

# Display sample data
print("\nSample Customer Data:")
print(customer_data.head())

print("\nSample Transaction Data:")
print(transaction_data.head())

# Basic statistics
print("\nCustomer Data Statistics:")
print(customer_data.describe())