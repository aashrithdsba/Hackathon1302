import pandas as pd
import numpy as np
from persona import build_persona_profiles
from Association import get_product_recommendations
from Popularity import get_trending_recommendations, calculate_trending_scores
from sequence_models import (
    train_prod2vec, 
    build_markov_transition_matrices, 
    get_similar_products_prod2vec,
    predict_next_state_markov
)
from Data_cleaning_FE import prepare_final_data

stocks_df = pd.read_csv('stocks.csv')
stocks_df['ProductID'] = stocks_df['ProductID'].astype(str)

def get_global_trends(data, date=None):
    scores = calculate_trending_scores(data, target_date=date)
    # Aggregate to global level
    # Ensure ProductID is string for consistency
    scores['ProductID'] = scores['ProductID'].astype(str)
    
    global_scores = scores.groupby('ProductID')['FinalScore'].sum().reset_index()
    global_scores = global_scores.sort_values('FinalScore', ascending=False)
    
    # Join with metadata for filtering
    # Ensure Catalog ProductID is also string
    catalog = data[['ProductID', 'FamilyLevel2']].drop_duplicates('ProductID').copy()
    catalog['ProductID'] = catalog['ProductID'].astype(str)
    
    return global_scores.merge(catalog, on='ProductID', how='left').head(100)
    

def get_final_recommendations(client_id, persona, final_data, models_dict, stocks_df, top_n=5):
    """
    The 'Conductor' logic: Routes each ClientID to a specific blend of engines 
    with a mandatory stock check based on StoreCountry.
    """
    
    # --- 1. STOCK PRE-PROCESSING ---
    user_info = final_data[final_data['ClientID'] == client_id]
    
    country = "Unknown"
    available_prods = set()
    
    if user_info.empty:
        # Fallback using global stock
        available_prods = set(stocks_df[stocks_df['Quantity'] > 0]['ProductID'].unique())
        print(f"DEBUG: Client {client_id} not found. Using global stock.")
    else:
        user_country = user_info['ClientCountry'].iloc[-1]
        country = user_country
        available_prods = set(stocks_df[
            (stocks_df['StoreCountry'] == user_country) & 
            (stocks_df['Quantity'] > 0)
        ]['ProductID'].unique())
        
    print(f"DEBUG: Client={client_id}, Persona='{persona}', Country='{country}'")
    print(f"DEBUG: Stock Avail={len(available_prods)} items")

    def apply_stock_filter(prod_list):
        return [p for p in prod_list if p in available_prods]
    
    # Helper to clean and backfill recommendations
    def finalize_list(primary_recs, fallback_pool, needed=top_n):
        # Dedupe primary
        seen = set()
        final = []
        for x in primary_recs:
            if x not in seen:
                final.append(x)
                seen.add(x)
        
        # Backfill if needed
        if len(final) < needed:
            for x in fallback_pool:
                if x not in seen:
                    final.append(x)
                    seen.add(x)
                    if len(final) >= needed:
                        break
        return final[:needed]

    # Pre-calculate fallbacks (Stock Checked)
    pop_filtered = apply_stock_filter(models_dict['popular']['ProductID'].tolist())
    trend_filtered = apply_stock_filter(models_dict['trending']['ProductID'].tolist())
    global_fallback = pop_filtered + trend_filtered # Combined pool

    # --- 2. ROUTING LOGIC ---
    
    # A. STATIC / COLD START
    if persona in ["No Transactions", "No Transactions (New)", "Very Frequent (>15)"]:
        print("DEBUG: Hitting Static Path")
        # Just return the top popular/trending items
        return finalize_list([], global_fallback)

    # B. INACTIVE (Reactivation)
    if "Inactive" in persona:
        print("DEBUG: Hitting Inactive Path")
        last_cat = user_info['FamilyLevel2'].iloc[-1]
        # Recommend popular items FROM THE SAME CATEGORY
        cat_recs = models_dict['popular'][models_dict['popular']['FamilyLevel2'] == last_cat]['ProductID'].tolist()
        valid_cat_recs = apply_stock_filter(cat_recs)
        
        print(f"DEBUG: Category Recs Found: {len(valid_cat_recs)}")
        return finalize_list(valid_cat_recs, global_fallback)

    # C. LOYAL / TOP (Association Focused)
    if "Loyal" in persona or "Top" in persona:
        print("DEBUG: Hitting Loyal Path")
        last_prod = str(user_info['ProductID'].iloc[-1])
        rules = models_dict.get('associations_pid', models_dict.get('associations'))
        
        matcher = rules['antecedents'].apply(lambda x: last_prod in x)
        matching_rules = rules[matcher]
        
        assoc_recs = []
        for c in matching_rules['consequents']:
            assoc_recs.extend(list(c))
            
        valid_assoc = apply_stock_filter(assoc_recs)
        print(f"DEBUG: Assoc Recs Found: {len(valid_assoc)}")
        
        return finalize_list(valid_assoc, global_fallback)

    # D. MID FREQ (AI Blend)
    if "Mid Freq" in persona:
        print("DEBUG: Hitting Mid Freq Path")
        last_prod = str(user_info['ProductID'].iloc[-1])
        last_fam = user_info['FamilyLevel2'].iloc[-1]
        
        # Prod2Vec
        try:
            p2v_raw = [res[0] for res in get_similar_products_prod2vec(models_dict['w2v'], last_prod, top_n=10)]
        except: p2v_raw = []
        
        # Markov
        try:
            matrix = models_dict['markov'].get(country, {})
            mkv_raw = [m[0] for m in predict_next_state_markov(matrix, last_fam, top_k=10)]
        except: mkv_raw = []
        
        # Association
        try:
            rules = models_dict.get('associations_pid', models_dict.get('associations'))
            matcher = rules['antecedents'].apply(lambda x: last_prod in x)
            matching_rules = rules[matcher]
            assoc_raw = []
            for c in matching_rules['consequents']:
                assoc_raw.extend(list(c))
        except: assoc_raw = []

        # Filter
        p2v_valid = apply_stock_filter(p2v_raw)
        mkv_valid = apply_stock_filter(mkv_raw)
        assoc_valid = apply_stock_filter(assoc_raw)
        
        print(f"DEBUG: P2V={len(p2v_valid)}, Markov={len(mkv_valid)}, Assoc={len(assoc_valid)}")
        
        # Blend Strategy: Interleave them
        # P2V, Mkv, Assoc, P2V, Mkv...
        blended = []
        max_len = max(len(p2v_valid), len(mkv_valid), len(assoc_valid))
        for i in range(max_len):
            if i < len(p2v_valid): blended.append(p2v_valid[i])
            if i < len(mkv_valid): blended.append(mkv_valid[i])
            if i < len(assoc_valid): blended.append(assoc_valid[i])
            
        return finalize_list(blended, global_fallback)

    # E. FALLBACK
    print(f"DEBUG: Hitting Generic Fallback (Persona '{persona}')")
    return finalize_list([], global_fallback)


def run_pipeline():
    # A. PREPARE DATA & STOCKS
    print("🧹 Cleaning data and loading stocks...")
    final_data = prepare_final_data('clients.csv', 'products.csv', 'stores.csv', 'transactions.csv')
    stocks_df = pd.read_csv('stocks.csv')
    stocks_df['ProductID'] = stocks_df['ProductID'].astype(str)
    
    # Dynamic Date for Trending logic
    latest_date_str = pd.to_datetime(final_data['SaleTransactionDate']).max().strftime('%Y-%m-%d')
    

    clients = pd.read_csv('clients.csv') 
    transactions = pd.read_csv('transactions.csv')
    
    print("👥 Categorizing users into personas...")
    persona_df = build_persona_profiles(clients, transactions)
    
    # B. TRAIN ALL MODELS 
    print("🎻 Training all models...")
    models_dict = {
        'w2v': train_prod2vec(final_data),
        'markov': build_markov_transition_matrices(final_data, state_col='FamilyLevel2'),
        'associations_pid': get_product_recommendations(final_data, item_col='ProductID'),
        'associations_fam': get_product_recommendations(final_data, item_col='FamilyLevel2'),
        'popular': get_global_trends(final_data), 
        'trending': get_global_trends(final_data, date=latest_date_str) 
    }
    
    # C. EXECUTE RECOMMENDATIONS
    # Example: Get recs for a specific client
    sample_client = persona_df.iloc[0]
    
    print(f"🚀 Generating In-Stock Recs for Client: {sample_client['ClientID']} ({sample_client['Persona']})")
    
    recommendations = get_final_recommendations(
        sample_client['ClientID'], 
        sample_client['Persona'], 
        final_data, 
        models_dict,
        stocks_df # Added stocks here
    )
    
    print(f"✅ Final Result: {recommendations}")

if __name__ == "__main__":
    run_pipeline()