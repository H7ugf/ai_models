import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Enable Arabic text in matplotlib
plt.rcParams['font.family'] = 'Arial'

# Read the dataset
df = pd.read_csv('C:/Users/pc/Desktop/homowrk_data_mining/aproir/Clean_Dataset.csv/Clean_Dataset.csv')

# Create transactions-like data
def prepare_transactions():
    # Create transactions by combining relevant features
    transactions = df.apply(lambda row: {
        f"airline_{row['airline']}", 
        f"class_{row['class']}", 
        f"stops_{row['stops']}", 
        f"source_{row['source_city']}", 
        f"dest_{row['destination_city']}",
        f"departure_{row['departure_time']}",
        f"arrival_{row['arrival_time']}"
    }, axis=1).tolist()
    return transactions

def plot_top_items(transactions, n=10):
    # Get all items
    all_items = [item for transaction in transactions for item in transaction]
    item_counts = Counter(all_items)
    top_items = item_counts.most_common(n)
    
    # Unzip the tuples
    items, counts = zip(*top_items)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(items), palette='viridis')
    plt.xlabel('عدد مرات التكرار')
    plt.ylabel('العناصر')
    plt.title('أكثر 10 عناصر تكراراً في الرحلات')
    plt.tight_layout()
    plt.savefig('top_items.png')
    plt.close()
    return dict(top_items)

def plot_transaction_distribution(transactions):
    transaction_lengths = [len(t) for t in transactions]
    plt.figure(figsize=(10, 6))
    sns.histplot(transaction_lengths, bins=range(1, max(transaction_lengths)+2), kde=True, color='skyblue')
    plt.xlabel('عدد العناصر في الرحلة')
    plt.ylabel('عدد الرحلات')
    plt.title('توزيع عدد العناصر في الرحلات')
    plt.tight_layout()
    plt.savefig('transaction_distribution.png')
    plt.close()

def plot_price_analysis():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50)
    plt.title('توزيع أسعار الرحلات')
    plt.xlabel('السعر')
    plt.ylabel('عدد الرحلات')
    plt.savefig('price_distribution.png')
    plt.close()

def plot_association_rules(rules):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x="support", y="confidence", 
                   size="lift", hue="lift", palette="cool", 
                   sizes=(40, 400))
    plt.title("قواعد الترابط: الدعم مقابل الثقة")
    plt.xlabel("الدعم")
    plt.ylabel("الثقة")
    plt.grid(True)
    plt.savefig('association_rules_scatter.png')
    plt.close()

def recommend_flights(features, rules):
    recommendations = set()
    features_set = set(features)
    for _, row in rules.iterrows():
        if row['antecedents'].issubset(features_set):
            recommendations |= row['consequents']
    return recommendations

def main():
    print("بدء التحليل...")
    
    # Prepare transactions
    transactions = prepare_transactions()
    
    # Plot top items
    print("\nتحليل العناصر الأكثر تكراراً...")
    top_items = plot_top_items(transactions)
    print("\nأكثر 10 عناصر تكراراً:")
    for item, count in top_items.items():
        print(f"{item}: {count}")
    
    # Plot transaction distribution
    print("\nتحليل توزيع العناصر في الرحلات...")
    plot_transaction_distribution(transactions)
    
    # Prepare data for Apriori
    print("\nتحضير البيانات لتحليل Apriori...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets and rules
    print("\nتوليد قواعد الترابط...")
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    rules = rules[rules['lift'] >= 1]
    
    # Save results
    rules.to_csv('association_rules.csv', index=False)
    frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
    
    # Create visualizations
    print("\nإنشاء الرسومات البيانية...")
    plot_price_analysis()
    plot_association_rules(rules)
    
    # Print top rules
    print("\nأفضل 5 قواعد ترابط:")
    top_rules = rules.sort_values('lift', ascending=False).head()
    print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
    # Example recommendation
    print("\nمثال على التوصيات:")
    example_features = {'airline_SpiceJet', 'class_Economy', 'source_Delhi'}
    recommendations = recommend_flights(example_features, rules)
    print(f"التوصيات للرحلة مع الخصائص {example_features}:")
    print(recommendations)

if __name__ == "__main__":
    main() 