import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

# =============================================================================
# HELPER FUNCTIONS (Internal Logic)
# =============================================================================

def _create_rfm_score(values, ascending=True):
    """
    Robust scoring function from Notebook Cell 6.
    Handles duplicate edges in qcut gracefully.
    """
    try:
        if ascending:
            score = pd.qcut(values, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        else:
            score = pd.qcut(values, q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

        # Map back to 1-5 if qcut dropped bins
        unique_cats = score.cat.categories
        if len(unique_cats) < 5:
            mapping = {}
            if ascending:
                for i, cat in enumerate(unique_cats): mapping[cat] = min(i + 1, 5)
            else:
                for i, cat in enumerate(unique_cats): mapping[cat] = max(5 - i, 1)
            score = score.map(mapping)
        return score.astype(int)
    except:
        # Fallback: Percentile method
        percentiles = values.rank(pct=True)
        if ascending:
            return pd.cut(percentiles, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True).astype(int)
        else:
            return pd.cut(percentiles, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[5, 4, 3, 2, 1], include_lowest=True).astype(int)

def _get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_final_data(clients_path, products_path, stores_path, transactions_path):
    """
    Exact emulation of EDA_FE (1).ipynb.
    Includes Sporting Events, Complex RFM, Category Affinity, and Product Velocity.
    """
    print("🚀 Starting PRO Data Pipeline...")
    warnings.filterwarnings("ignore")

    ## 1. LOAD & CLEAN
    #trans = pd.read_csv(transactions_path, parse_dates=['SaleTransactionDate'])


    # 1. Load the CSV normally
    trans = pd.read_csv(transactions_path, dtype={'ClientID': str, 'ProductID': str})
    print(trans.columns)
    # 2. Convert to datetime
    trans['SaleTransactionDate'] = pd.to_datetime(trans['SaleTransactionDate'], errors='coerce')

    # 3. STRIP TIMEZONE (The Crucial Step)
    # This converts '2023-06-06 00:00:00+00:00' to '2023-06-06 00:00:00'
    trans['SaleTransactionDate'] = trans['SaleTransactionDate'].dt.tz_localize(None)

    # 4. Final check: Sort by date so your Markov/Prod2Vec sequences are correct
    trans = trans.sort_values('SaleTransactionDate')

    prod = pd.read_csv(products_path, dtype={'ProductID': str})
    client = pd.read_csv(clients_path, dtype={'ClientID': str})
    store = pd.read_csv(stores_path)

    # --- FIX: REMOVE TIMEZONE INFO ---
    # This line solves the "Invalid comparison" TypeError
    if trans['SaleTransactionDate'].dt.tz is not None:
        trans['SaleTransactionDate'] = trans['SaleTransactionDate'].dt.tz_localize(None)

    # Basic Cleaning
    trans = trans.drop_duplicates()
    trans = trans[(trans['SalesNetAmountEuro'] >= 0) & (trans['Quantity'] > 0)]
    
    prod['Category'] = prod['Category'].fillna('Unknown')
    prod['Universe'] = prod['Universe'].fillna('Unisex')
    prod[['FamilyLevel1', 'FamilyLevel2']] = prod[['FamilyLevel1', 'FamilyLevel2']].fillna('Unknown')
    
    if 'Age' in client.columns: client = client.drop('Age', axis=1)
    client['ClientGender'] = client['ClientGender'].fillna('Unknown')
    client['ClientSegment'] = client['ClientSegment'].fillna('Regular')
    
    if 'ClientOptINEmail' in client.columns:
        client['HasEmail'] = (client['ClientOptINEmail'] == 1).astype(int)
    else:
        client['HasEmail'] = 0
        
    if 'ClientOptINPhone' in client.columns:
        client['HasPhone'] = (client['ClientOptINPhone'] == 1).astype(int)
    else:
        client['HasPhone'] = 0
        
    client['HasContact'] = ((client['HasEmail'] == 1) | (client['HasPhone'] == 1)).astype(int)

    # 2. MASTER MERGE
    master = trans.merge(
        prod[['ProductID', 'Category', 'FamilyLevel1', 'FamilyLevel2', 'Universe']], on='ProductID', how='left'
    ).merge(
        client[['ClientID', 'ClientSegment', 'ClientCountry', 'ClientGender', 'HasEmail', 'HasPhone', 'HasContact']], on='ClientID', how='left'
    ).merge(
        store[['StoreID', 'StoreCountry']], on='StoreID', how='left'
    )

    # Temporal Basics
    master['Year'] = master['SaleTransactionDate'].dt.year
    master['Month'] = master['SaleTransactionDate'].dt.month
    master['Quarter'] = master['SaleTransactionDate'].dt.quarter
    master['DayOfWeek'] = master['SaleTransactionDate'].dt.dayofweek
    master['DayOfMonth'] = master['SaleTransactionDate'].dt.day
    master['IsWeekend'] = master['DayOfWeek'].isin([5, 6]).astype(int)
    master['Season'] = master['Month'].apply(_get_season)
    
    for s in ['Winter', 'Spring', 'Summer', 'Fall']: 
        master[f'Is{s}'] = (master['Season'] == s).astype(int)
    for q in [1, 2, 3, 4]: 
        master[f'IsQ{q}'] = (master['Quarter'] == q).astype(int)

    # 3. ADVANCED EVENTS
    print("   Adding Events (Black Friday, Sports)...")
    
    master['IsBlackFriday'] = 0
    master['IsCyberMonday'] = 0
    for year in master['Year'].unique():
        nov_1 = pd.Timestamp(f'{year}-11-01')
        first_friday = nov_1 + timedelta(days=(4 - nov_1.dayofweek) % 7)
        bf = first_friday + timedelta(weeks=3)
        cm = bf + timedelta(days=3)
        
        # Black Friday (Fri-Sun)
        master.loc[(master['SaleTransactionDate'] >= bf) & (master['SaleTransactionDate'] <= bf + timedelta(days=3)), 'IsBlackFriday'] = 1
        # Cyber Monday
        master.loc[(master['SaleTransactionDate'] >= cm) & (master['SaleTransactionDate'] <= cm), 'IsCyberMonday'] = 1

    master['IsChristmasSeason'] = ((master['Month'] == 12) & (master['DayOfMonth'] <= 25)).astype(int)
    master['IsNewYearSeason'] = (((master['Month'] == 12) & (master['DayOfMonth'] >= 26)) | ((master['Month'] == 1) & (master['DayOfMonth'] <= 7))).astype(int)
    master['IsBackToSchool'] = (((master['Month'] == 8) & (master['DayOfMonth'] >= 15)) | ((master['Month'] == 9) & (master['DayOfMonth'] <= 15))).astype(int)
    master['IsSummerSale'] = (((master['Month'] == 6) & (master['DayOfMonth'] >= 15)) | (master['Month'] == 7)).astype(int)
    master['IsHoliday'] = ((master['IsWeekend'] == 1) | (master['IsChristmasSeason'] == 1) | (master['IsNewYearSeason'] == 1) | (master['IsBlackFriday'] == 1)).astype(int)

    # Major Sporting Events
    master['IsMajorSportingEvent'] = 0
    sporting_events = [
        ('2022-11-20', '2022-12-18', ['All']), # World Cup
        ('2024-06-14', '2024-07-14', ['FR', 'ES', 'IT', 'DE', 'GB', 'UK']), # Euro
        ('2024-07-26', '2024-08-11', ['All']), # Olympics
        ('2023-06-10', '2023-06-10', ['FR', 'ES', 'IT', 'DE', 'GB', 'UK']), # CL Final
        ('2024-06-01', '2024-06-01', ['FR', 'ES', 'IT', 'DE', 'GB', 'UK']), # CL Final
        ('2023-02-12', '2023-02-12', ['US']), # Super Bowl
        ('2024-02-11', '2024-02-11', ['US']), # Super Bowl
    ]
    
    for start, end, countries in sporting_events:
        # Convert start/end to naive timestamps for safe comparison
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        mask = (master['SaleTransactionDate'] >= start_ts) & (master['SaleTransactionDate'] <= end_ts)
        if 'All' not in countries:
            mask = mask & master['ClientCountry'].isin(countries)
        master.loc[mask, 'IsMajorSportingEvent'] = 1

    # 4. ROBUST CUSTOMER RFM
    print("   Calculating RFM...")
    ref_date = master['SaleTransactionDate'].max() + timedelta(days=1)
    
    rfm = master.groupby('ClientID').agg({
        'SaleTransactionDate': [lambda x: (ref_date - x.max()).days, 'min', 'max'],
        'SalesNetAmountEuro': ['sum', 'mean', 'std'],
        'Quantity': ['sum'],
        'ClientID': 'count', # Frequency
        'IsWeekend': 'mean'
    }).reset_index()
    
    rfm.columns = ['ClientID', 'Recency', 'FirstPurchase', 'LastPurchase', 'TotalSpent', 'AvgOrderValue', 'StdOrderValue', 'TotalQuantity', 'Frequency', 'WeekendShopperRatio']
    
    # Apply Robust Scoring
    rfm['RecencyScore'] = _create_rfm_score(rfm['Recency'], ascending=False)
    rfm['FrequencyScore'] = _create_rfm_score(rfm['Frequency'], ascending=True)
    rfm['MonetaryScore'] = _create_rfm_score(rfm['TotalSpent'], ascending=True)
    rfm['RFM_Score'] = rfm['RecencyScore'] + rfm['FrequencyScore'] + rfm['MonetaryScore']
    
    # Derived Customer Features
    rfm['CustomerLifetimeDays'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days
    rfm['PriceVolatility'] = (rfm['StdOrderValue'] / rfm['AvgOrderValue'].clip(lower=1)).fillna(0)
    
    # Segments
    rfm['IsHighValue'] = (rfm['MonetaryScore'] >= 4).astype(int)
    rfm['IsFrequentBuyer'] = (rfm['FrequencyScore'] >= 4).astype(int)
    rfm['IsRecentBuyer'] = (rfm['RecencyScore'] >= 4).astype(int)
    rfm['IsAtRisk'] = ((rfm['RecencyScore'] <= 2) & (rfm['FrequencyScore'] >= 3)).astype(int)

    master = master.merge(rfm.drop(['FirstPurchase', 'LastPurchase'], axis=1), on='ClientID', how='left')

    # 5. PRODUCT FEATURES
    print("   Calculating Product Features...")
    prod_feat = master.groupby('ProductID').agg({
        'ClientID': 'nunique',
        'SalesNetAmountEuro': ['sum'],
        'Quantity': ['sum'],
        'ProductID': 'count',
        'SaleTransactionDate': ['min', 'max']
    }).reset_index()
    prod_feat.columns = ['ProductID', 'ProductUniqueCustomers', 'ProductTotalRevenue', 'ProductTotalUnitsSold', 'ProductTransactionCount', 'FirstSale', 'LastSale']
    
    prod_feat['ProductDaysActive'] = (prod_feat['LastSale'] - prod_feat['FirstSale']).dt.days
    prod_feat['ProductSalesVelocity'] = prod_feat['ProductTotalUnitsSold'] / prod_feat['ProductDaysActive'].clip(lower=1)
    
    # Tiers
    prod_feat['IsTopProduct'] = (prod_feat['ProductTotalRevenue'] > prod_feat['ProductTotalRevenue'].quantile(0.8)).astype(int)
    prod_feat['IsFastMover'] = (prod_feat['ProductSalesVelocity'] > prod_feat['ProductSalesVelocity'].quantile(0.8)).astype(int)
    
    master = master.merge(prod_feat.drop(['FirstSale', 'LastSale'], axis=1), on='ProductID', how='left')

    # 6. CATEGORY AFFINITY
    print("   Calculating Category Affinity...")
    cat_spend = master.groupby(['ClientID', 'Category'])['SalesNetAmountEuro'].sum().reset_index()
    fav_cat = cat_spend.loc[cat_spend.groupby('ClientID')['SalesNetAmountEuro'].idxmax()]
    fav_cat = fav_cat[['ClientID', 'Category']].rename(columns={'Category': 'FavoriteCategory'})
    
    master = master.merge(fav_cat, on='ClientID', how='left')
    master['IsFavoriteCategory'] = (master['Category'] == master['FavoriteCategory']).astype(int)
    master['IsCrossSellOpportunity'] = (master['Category'] != master['FavoriteCategory']).astype(int)

    # 7. DERIVED TRANSACTION FEATURES
    print("   Finalizing Transaction Features...")
    master['PricePerUnit'] = master['SalesNetAmountEuro'] / master['Quantity'].clip(lower=1)
    
    # Discount Logic
    avg_prices = master.groupby('ProductID')['PricePerUnit'].mean()
    master['AvgPricePerUnit'] = master['ProductID'].map(avg_prices)
    master['IsPotentialDiscount'] = (master['PricePerUnit'] < (master['AvgPricePerUnit'] * 0.8)).astype(int)
    master['DiscountPercentage'] = ((1 - (master['PricePerUnit'] / master['AvgPricePerUnit'].clip(lower=0.01))) * 100).clip(0, 100)
    
    # Premium / Bulk
    master['IsPremiumPurchase'] = (master['PricePerUnit'] > (master['AvgPricePerUnit'] * 1.2)).astype(int)
    typical_qty = master.groupby('ProductID')['Quantity'].median()
    master['TypicalQuantity'] = master['ProductID'].map(typical_qty)
    master['IsBulkPurchase'] = (master['Quantity'] >= (master['TypicalQuantity'] * 3).clip(lower=5)).astype(int)
    
    # Days Since Last Purchase (Complex Diff)
    master = master.sort_values(['ClientID', 'SaleTransactionDate'])
    master['DaysSinceLastPurchase'] = master.groupby('ClientID')['SaleTransactionDate'].diff().dt.days.fillna(9999)

    # Pareto (Top 20% Revenue)
    total_rev = master['SalesNetAmountEuro'].sum()
    master_sorted = master.sort_values('SalesNetAmountEuro', ascending=False)
    master_sorted['CumulativeRevenue'] = master_sorted['SalesNetAmountEuro'].cumsum()
    master_sorted['IsTop20PercentRevenue'] = (master_sorted['CumulativeRevenue'] <= (total_rev * 0.8)).astype(int)
    
    # Map back cleanly
    master['IsTop20PercentRevenue'] = master_sorted['IsTop20PercentRevenue'].sort_index()

    # Location features
    if 'StoreCountry' in master.columns and 'ClientCountry' in master.columns:
        master['IsCrossBorder'] = (master['StoreCountry'] != master['ClientCountry']).astype(int)
        master['IsLocalPurchase'] = (master['StoreCountry'] == master['ClientCountry']).astype(int)

    print(f"✅ Pipeline Complete! Shape: {master.shape}")
    return master

if __name__ == "__main__":
    print("PRO Data Pipeline loaded.")