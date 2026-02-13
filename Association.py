import pandas as pd
from collections import defaultdict
from itertools import combinations

def get_product_recommendations(final_data, min_support=0.0005, item_col='ProductID'):
    """
    Generates product association rules based on co-occurrence.
    Returns a DataFrame compatible with mlxtend format (antecedents, consequents, etc.)
    using specified column (ProductID or FamilyLevel2).
    """
    
    # 1. Ensure BasketID exists
    df = final_data.copy()
    if "BasketID" not in df.columns:
        # Construct BasketID from Client + Store + Date (if avail) to capture actual "trips"
        # rather than lifetime history.
        cols = []
        if "ClientID" in df.columns: cols.append(df["ClientID"].astype(str))
        if "SaleTransactionDate" in df.columns: cols.append(df["SaleTransactionDate"].astype(str))
        if "StoreID" in df.columns: cols.append(df["StoreID"].astype(str))
        
        if cols:
            df["BasketID"] = cols[0]
            for c in cols[1:]:
                df["BasketID"] = df["BasketID"] + "_" + c
        else:
             df["BasketID"] = df.index.astype(str)

    # 2. Group by Basket -> Items
    # Ensure item_col is string
    df[item_col] = df[item_col].astype(str)
    
    # Optimized aggregation (set is faster than lambda x: sorted(list(set(x))))
    basket_items = df.groupby("BasketID")[item_col].agg(set).tolist()
    
    # Filter out single-item baskets (no pairs possible)
    basket_items = [sorted(list(b)) for b in basket_items if len(b) > 1]
    
    total_baskets = len(basket_items)
    min_support_count = total_baskets * min_support

    # 3. Count Items and Pairs
    pair_counts = defaultdict(int)
    item_counts = defaultdict(int)

    for items in basket_items:
        # Count individual items
        for i in items:
            item_counts[i] += 1
            
        # Count pairs
        for a, b in combinations(items, 2):
            pair_counts[(a, b)] += 1
            pair_counts[(b, a)] += 1 # Symmetric for AB vs BA rules

    # 4. Generate Rules
    rows = []
    
    # Pre-calculate item supports
    item_support = {k: v/total_baskets for k, v in item_counts.items()}

    for (ant, cons), count in pair_counts.items():
        if count < min_support_count:
            continue
            
        support = count / total_baskets
        ant_support = item_support[ant]
        cons_support = item_support[cons]
        
        confidence = support / ant_support
        lift = confidence / cons_support
        
        rows.append({
            'antecedents': frozenset([ant]),
            'consequents': frozenset([cons]),
            'support': support,
            'confidence': confidence,
            'lift': lift
        })

    rules_df = pd.DataFrame(rows)
    
    if rules_df.empty:
         return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
         
    return rules_df.sort_values('lift', ascending=False)

if __name__ == "__main__":
    print("Product Recommender library loaded.")