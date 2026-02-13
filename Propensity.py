import pandas as pd
import numpy as np

def build_customer_affinity_matrix(df, current_date):
    """
    Builds a customer affinity profile based on their purchase history.
    Calculates the 'Share of Wallet' for each category (FamilyLevel1).
    
    Args:
        df (pd.DataFrame): The transaction data (can be filtered by country).
        current_date (str/datetime): The reference date for recency calculations.
        
    Returns:
        pd.DataFrame: A matrix of customer affinity scores.
    """
    # Ensure date format
    df = df.copy()
    df['SaleTransactionDate'] = pd.to_datetime(df['SaleTransactionDate'])
    current_date = pd.to_datetime(current_date)

    # --- A. GLOBAL STATS (Total Spend per Customer) ---
    # We need this to calculate "Share of Wallet"
    global_stats = df.groupby('ClientID').agg(
        TotalSpend=('SalesNetAmountEuro', 'sum'),
        LastPurchaseDate=('SaleTransactionDate', 'max')
    ).reset_index()

    global_stats['DaysSinceLastPurchase'] = (current_date - global_stats['LastPurchaseDate']).dt.days

    # --- B. CATEGORY STATS (Spend per Category) ---
    category_stats = df.groupby(['ClientID', 'FamilyLevel1']).agg(
        CategorySpend=('SalesNetAmountEuro', 'sum'),
        PurchaseCount=('SaleTransactionDate', 'count')
    ).reset_index()

    # --- C. MERGE & CALCULATE AFFINITY ---
    affinity = pd.merge(category_stats, global_stats, on='ClientID', how='left')

    # Share of Wallet: How much of their total spend goes to this category?
    affinity['ShareOfWallet'] = affinity['CategorySpend'] / affinity['TotalSpend']

    # Normalized Affinity Score (0 to 1)
    # Combination of "How much they spend" and "How often they buy"
    affinity['AffinityScore'] = (affinity['ShareOfWallet'] * 0.7) + \
                                (affinity['PurchaseCount'] / affinity['PurchaseCount'].max() * 0.3)
    
    return affinity

def calculate_propensity_score(affinity_df, target_product_family, recency_weight=0.5):
    """
    Calculates a final propensity score for a specific target family.
    """
    # Filter for the specific category
    target_affinity = affinity_df[affinity_df['FamilyLevel1'] == target_product_family].copy()
    
    if target_affinity.empty:
        return pd.DataFrame()

    # Recency Decay: The longer they haven't bought, the lower the score
    target_affinity['RecencyFactor'] = 1 / np.log1p(target_affinity['DaysSinceLastPurchase'] + 1)
    
    target_affinity['PropensityScore'] = (target_affinity['AffinityScore'] * (1 - recency_weight)) + \
                                         (target_affinity['RecencyFactor'] * recency_weight)
    
    return target_affinity.sort_values('PropensityScore', ascending=False)

def get_propensity_targeting(df, current_date, target_family=None, target_product_id=None, target_country=None):
    """
    Main entry point: Generates targeted lists for a specific Family or Cross-Sell.
    
    Args:
        df (pd.DataFrame): The raw transaction data.
        current_date (str): Reference date for analysis.
        target_family (str, optional): The product family to target (e.g., 'Shoes').
        target_product_id (str, optional): Specific Product ID for cross-sell campaigns.
        target_country (str, optional): Filter customers/sales by StoreCountry (e.g., 'FRA').
        
    Returns:
        pd.DataFrame: A list of customers sorted by PropensityScore.
    """
    # 0. Filter by Country if requested
    # We keep a reference to full_df for global product lookups if needed
    full_df = df.copy()
    
    if target_country:
        if 'StoreCountry' not in df.columns:
            raise ValueError("Dataframe is missing 'StoreCountry' column, cannot filter.")
        calc_df = df[df['StoreCountry'] == target_country].copy()
        
        if calc_df.empty:
            print(f"Warning: No data found for Country {target_country}")
            return pd.DataFrame()
    else:
        calc_df = full_df

    # 1. Build Base Profiles (on filtered data)
    affinity_matrix = build_customer_affinity_matrix(calc_df, current_date)

    if affinity_matrix.empty:
        return pd.DataFrame()

    # 2. Case A: Category Targeting (Generic)
    if target_family and not target_product_id:
        scores = calculate_propensity_score(affinity_matrix, target_family)
        if scores.empty: return pd.DataFrame()
        return scores[['ClientID', 'TotalSpend', 'DaysSinceLastPurchase', 'PropensityScore']]

    # 3. Case B: Cross-Sell Targeting (Specific Product)
    if target_product_id:
        target_id_str = str(target_product_id)
        
        # A. Find the family of this product
        # Look in full_df first (in case product isn't sold in target country yet)
        product_info = full_df[full_df['ProductID'].astype(str) == target_id_str]
        
        if product_info.empty:
            print(f"Product {target_product_id} not found in transaction history.")
            return pd.DataFrame()
            
        derived_family = product_info.iloc[0]['FamilyLevel1']
        
        # B. Get Scores for that Family (using Local/Filtered data)
        scores = calculate_propensity_score(affinity_matrix, derived_family)
        
        if scores.empty: return pd.DataFrame()
        
        # C. Exclude customers who already bought this specific product
        # (We check Global history to ensure we don't sell them what they own, 
        # even if they bought it in another country)
        existing_buyers = full_df[full_df['ProductID'].astype(str) == target_id_str]['ClientID'].unique()
        
        # Filter
        cross_sell_candidates = scores[~scores['ClientID'].isin(existing_buyers)]
        
        return cross_sell_candidates[['ClientID', 'PropensityScore', 'DaysSinceLastPurchase']]
    
    return pd.DataFrame()

if __name__ == "__main__":
    print("Propensity Scorer library loaded (Version 2.0 - Country Support added).")