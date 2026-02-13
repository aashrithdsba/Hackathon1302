import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_persona_profiles(clients_df, transactions_df):
    """
    Classifies customers into 6 specific categories based on Transaction Count and ClientSegment.
    
    Args:
        clients_df (pd.DataFrame): Clients table (must contain 'ClientID' and 'ClientSegment').
        transactions_df (pd.DataFrame): Transactions table (must contain 'ClientID').
        
    Returns:
        pd.DataFrame: Merged DataFrame with a new 'Persona' column.
    """
    print("👥 Building Customer Personas (6-Category Logic)...")
    
    # 1. Calculate Transaction Counts per Client
    # We only need the count of transactions here
    trans_counts = transactions_df.groupby('ClientID').size().reset_index(name='TransactionCount')
    
    # 2. Merge with Client Table
    # We use 'left' merge on clients to keep Prospects who have 0 transactions
    df = pd.merge(clients_df, trans_counts, on='ClientID', how='left')
    
    # Fill NaN counts with 0 (for those with no transactions)
    df['TransactionCount'] = df['TransactionCount'].fillna(0).astype(int)
    
    # 3. Apply the 6-Category Classification Rules
    df['Persona'] = df.apply(_classify_specific_persona, axis=1)
    
    # Check the distribution
    print("\n📊 Persona Distribution:")
    print(df['Persona'].value_counts())
    
    return df

def _classify_specific_persona(row):
    """
    Implements the specific user-defined logic:
    1. Overall: No transactions (Count == 0)
    2. Overall: >15 transactions
    3. INACTIVE_1Y: 3 to 15 (Mid) vs 1 to 2 (Low)
    4. LOYAL/TOP: 3 to 15 (Mid) vs 1 to 2 (Low)
    """
    segment = str(row['ClientSegment']).upper()
    count = row['TransactionCount']

    # --- Rule 1: Overall - No transactions ---
    if count == 0:
        return "No Transactions"

    # --- Rule 2: Overall - >15 transactions ---
    if count > 15:
        return "Very Frequent (>15)"

    # --- Rule 3 & 4: INACTIVE_1Y ---
    # (Here count is between 1 and 15)
    if segment == 'INACTIVE_1Y':
        if count > 2:  # Covers 3, 4, ... 15
            return "Inactive (Mid Freq)"
        else:          # Covers 1, 2
            return "Inactive (Low Freq)"

    # --- Rule 5 & 6: LOYAL / TOP ---
    if segment in ['LOYAL', 'TOP']:
        if count > 2:  # Covers 3, 4, ... 15
            return "Loyal/Top (Mid Freq)"
        else:          # Covers 1, 2
            return "Loyal/Top (Low Freq)"

    # Fallback for other segments (e.g. PROSPECT) with 1-15 transactions
    return "Other (Active)"

def plot_persona_distribution(persona_df):
    """
    Simple bar chart of the new personas.
    """
    plt.figure(figsize=(10, 6))
    order = [
        "No Transactions", 
        "Very Frequent (>15)", 
        "Loyal/Top (Mid Freq)", 
        "Loyal/Top (Low Freq)", 
        "Inactive (Mid Freq)", 
        "Inactive (Low Freq)"
    ]
    # Filter order to only exist in data
    order = [o for o in order if o in persona_df['Persona'].unique()]
    
    sns.countplot(y='Persona', data=persona_df, order=order, palette='viridis')
    plt.title('Customer Segments (6 Categories)')
    plt.xlabel('Number of Clients')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Customer Persona Library (6-Category Logic - Revised) loaded.")