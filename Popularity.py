import pandas as pd
import numpy as np

def get_trending_recommendations(final_data, target_store, target_country, top_n=5, target_date=None):
    """
    Main entry point: Generates top trending recommendations for a specific store/country.
    It handles the full pipeline: Preprocessing -> Scoring -> Recommendation filtering.
    
    Args:
        final_data (pd.DataFrame): The raw merged transaction data.
        target_store (str): The StoreID to get recommendations for.
        target_country (str): The Country Code (e.g., 'FRA', 'BRA').
        top_n (int): Number of recommendations to return.
        target_date (str/datetime, optional): The 'current' date for analysis. 
                                              Defaults to the max date in data.
                                              
    Returns:
        pd.DataFrame: Top N recommendations with columns [Product, Family, Price, Score]
        str: The source level used (Store, Country, or Global)
    """
    # 1. Preprocess & Score the Data
    scores = calculate_trending_scores(final_data, target_date)
    
    # 2. Filter Recommendations for the specific target
    recs, source = _filter_recommendations(scores, target_store, target_country, top_n)
    
    return recs, source

def calculate_trending_scores(data, target_date=None):
    """
    Calculates robust trending scores based on Recency (70%) and Seasonality (30%).
    """
    # --- A. PREPROCESSING ---
    df_calc = data.copy()
    
    # Ensure Date Format
    df_calc['SaleTransactionDate'] = pd.to_datetime(df_calc['SaleTransactionDate'])
    
    # Set Target Date (Default to max date in data if not provided)
    if target_date is None:
        target_date = df_calc['SaleTransactionDate'].max()
    else:
        target_date = pd.Timestamp(target_date)

    # Calculate PricePerUnit if missing
    if 'PricePerUnit' not in df_calc.columns:
        df_calc['PricePerUnit'] = df_calc['SalesNetAmountEuro'] / df_calc['Quantity']

    # Filter Noise (Low Value & Bulk Orders) - strict 10th/99th percentile
    price_floor = df_calc['PricePerUnit'].quantile(0.10)
    qty_ceiling = df_calc['Quantity'].quantile(0.99)
    df_calc = df_calc[
        (df_calc['PricePerUnit'] >= price_floor) & 
        (df_calc['Quantity'] <= qty_ceiling)
    ].copy()

    # Ensure String Types for Keys
    for col in ['StoreID', 'StoreCountry', 'ProductID', 'FamilyLevel1']:
        if col in df_calc.columns:
            df_calc[col] = df_calc[col].astype(str)

    # --- B. CREATE CATALOG ---
    # Reference table for Price & Family (handles missing history)
    catalog = df_calc.sort_values('SaleTransactionDate').groupby('ProductID').agg({
        'FamilyLevel1': 'last',
        'PricePerUnit': 'mean' 
    }).reset_index()

    # --- C. RECENCY SCORE (Last 60 Days) ---
    recent_cutoff = target_date - pd.Timedelta(days=60)
    recent_df = df_calc[df_calc['SaleTransactionDate'] >= recent_cutoff].copy()

    # Time Decay: Newer sales weighted higher
    recent_df['days_ago'] = (target_date - recent_df['SaleTransactionDate']).dt.days
    recent_df['recency_weight'] = 1 / (1 + (recent_df['days_ago'] * 0.1))

    recent_scores = recent_df.groupby(['StoreID', 'StoreCountry', 'ProductID'])['recency_weight'].sum().reset_index()
    recent_scores.rename(columns={'recency_weight': 'recent_score'}, inplace=True)

    # --- D. SEASONAL SCORE (Same Quarter History) ---
    target_q = target_date.quarter
    
    # Strictly older than recent window (to capture true history)
    df_calc['Quarter'] = df_calc['SaleTransactionDate'].dt.quarter
    history_df = df_calc[
        (df_calc['Quarter'] == target_q) & 
        (df_calc['SaleTransactionDate'] < recent_cutoff)
    ].copy()

    history_scores = history_df.groupby(['StoreID', 'StoreCountry', 'ProductID'])['Quantity'].sum().reset_index()
    history_scores.rename(columns={'Quantity': 'seasonal_score'}, inplace=True)

    # --- E. MERGE & WEIGHT ---
    final_df = pd.merge(recent_scores, history_scores, on=['StoreID', 'StoreCountry', 'ProductID'], how='outer').fillna(0)
    
    # Attach Catalog details
    final_df = pd.merge(final_df, catalog, on='ProductID', how='left')
    final_df['FamilyLevel1'] = final_df['FamilyLevel1'].fillna('Unknown')
    final_df['PricePerUnit'] = final_df['PricePerUnit'].fillna(0)

    # Weighted Score: 70% Recent Trend, 30% Historical Seasonality
    final_df['FinalScore'] = (final_df['recent_score'] * 0.7) + (final_df['seasonal_score'] * 0.3)
    
    return final_df.sort_values('FinalScore', ascending=False)

def _filter_recommendations(scores, target_store, target_country, top_n):
    """
    Internal helper: Applies Waterfall logic (Store -> Country -> Global) and Diversity filter.
    """
    # 1. Store Level
    candidates = scores[scores['StoreID'] == str(target_store)].copy()
    source = "Store Level"

    # 2. Country Fallback (if Store score too low)
    if candidates['FinalScore'].sum() < 10:
        candidates = scores[scores['StoreCountry'] == str(target_country)].copy()
        source = "Country Level"

        # 3. Global Fallback
        if candidates['FinalScore'].sum() < 10:
            candidates = scores.copy()
            source = "Global Level"

    candidates = candidates.sort_values('FinalScore', ascending=False)

    # 4. Diversity Filter (One item per Family)
    final_recs = []
    seen_families = set()

    for _, row in candidates.iterrows():
        if len(final_recs) >= top_n: break
        
        fam = str(row['FamilyLevel1'])
        if fam not in seen_families and fam != 'Unknown':
            final_recs.append({
                'Product': row['ProductID'],
                'Family': fam,
                'Price': f"€{row['PricePerUnit']:.2f}",
                'Score': round(row['FinalScore'], 2)
            })
            seen_families.add(fam)

    return pd.DataFrame(final_recs), source

if __name__ == "__main__":
    print("Trending Recommender library loaded.")