import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Final_Orchestra_2 import prepare_final_data, get_global_trends, get_final_recommendations
from sequence_models import (
    train_prod2vec, 
    build_markov_transition_matrices, 
    get_similar_products_prod2vec,
    predict_next_state_markov
)
from Association import get_product_recommendations as get_assoc_rules
from Popularity import calculate_trending_scores
from persona import build_persona_profiles

# --- PAGE CONFIG ---
# --- PAGE CONFIG ---
st.set_page_config(page_title="STRIDE Analytics", layout="wide", initial_sidebar_state="expanded")

# --- STYLING (From Dashboard.py) ---
st.markdown("""
<style>
    /* Main Background & Sidebar */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #e94560;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #ccc; }
    [data-testid="stMetricValue"] { font-size: 1.30rem; font-weight: 700; color: #ffffff; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #aaa;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #e94560 !important;
        color: #ffffff !important;
    }
    
    /* Global Text */
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, li, span, label, .stMarkdown { color: #e0e0e0; }
    .stCaption { color: #aaa !important; }
    
    /* Custom Cards */
    .country-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .country-card h3 { margin: 0 0 8px 0; color: #e94560 !important; font-size: 1.1rem; }
    .rec-badge {
        display: inline-block;
        background: rgba(233,69,96,0.15);
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 6px;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.82rem;
        color: #ff8fa3;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & LOGO ---
st.sidebar.image("stride_logo.png", use_container_width=True)
st.sidebar.markdown("### STRIDE Retail Engine")

# --- CONSTANTS & HELPERS ---
PALETTE = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#533483", "#2b9348", "#d62828", "#f77f00"]

PERSONA_STRATEGIES = {
    "No Transactions": {
        "strategy": "Welcome offer on trending products",
        "channel": "Email + Social Media",
        "urgency": "Medium",
    },
    "Inactive (Low Freq)": {
        "strategy": "Win-back with discount on last-purchased category",
        "channel": "Email + SMS + Retargeting",
        "urgency": "High",
    },
    "Inactive (Mid Freq)": {
        "strategy": "Re-engage with personalised product bundles",
        "channel": "Email + Push",
        "urgency": "High",
    },
    "Loyal/Top (Low Freq)": {
        "strategy": "Cross-sell via association rules to build basket",
        "channel": "Email + In-App",
        "urgency": "Moderate",
    },
    "Loyal/Top (Mid Freq)": {
        "strategy": "AI-powered next-product predictions + loyalty rewards",
        "channel": "Personalised Email + In-App + SMS",
        "urgency": "Nurture",
    },
    "Very Frequent (>15)": {
        "strategy": "VIP treatment, ambassador program, exclusive early access",
        "channel": "All Channels + Personal Manager",
        "urgency": "VIP",
    },
}

def fmt_eur(val):
    return f"€{val:,.0f}" if abs(val) >= 1000 else f"€{val:,.2f}"

def gender_label(g):
    return {"M": "Men", "F": "Women"}.get(g, "All")

# --- CACHED DATA LOADING ---
@st.cache_resource
def load_and_train_models():
    # 1. Load Data
    final_data = prepare_final_data('clients.csv', 'products.csv', 'stores.csv', 'transactions.csv')
    final_data['ProductID'] = final_data['ProductID'].astype(str)
    
    stocks_df = pd.read_csv('stocks.csv')
    stocks_df['ProductID'] = stocks_df['ProductID'].astype(str)
    latest_date_str = pd.to_datetime(final_data['SaleTransactionDate']).max().strftime('%Y-%m-%d')
    clients = pd.read_csv('clients.csv', dtype={'ClientID': str}) 
    transactions = pd.read_csv('transactions.csv', dtype={'ClientID': str, 'ProductID': str})
    
    # 2. Personas
    persona_df = build_persona_profiles(clients, transactions)

    # --- ENRICH PERSONA WITH SPEND & RECENCY ---
    # Calculate stats from Final Data (Cleaner) or Transactions
    # Using final_data grouped by ClientID
    
    # Ensure Date is datetime
    final_data['SaleTransactionDate'] = pd.to_datetime(final_data['SaleTransactionDate'])
    max_date = final_data['SaleTransactionDate'].max()
    
    cust_stats = final_data.groupby('ClientID').agg({
        'SalesNetAmountEuro': 'sum',
        'SaleTransactionDate': 'max',
        'Quantity': 'sum'
    }).rename(columns={
        'SalesNetAmountEuro': 'TotalSpent',
        'SaleTransactionDate': 'LastDate',
        'Quantity': 'TotalUnits'
    })
    
    cust_stats['Recency'] = (max_date - cust_stats['LastDate']).dt.days
    
    # Fav Category (Mode)
    # Optimization: Do this only if needed, can be slow. Let's do a simple one.
    fav_cat = final_data.groupby('ClientID')['Category'].agg(lambda x: x.mode()[0] if not x.mode().empty else "N/A")
    cust_stats['FavoriteCategory'] = fav_cat
    
    # Merge into Persona DF
    persona_df = persona_df.merge(cust_stats, on='ClientID', how='left')
    persona_df['AvgOrderValue'] = persona_df['TotalSpent'] / persona_df['TransactionCount']
    
    # Fill NaNs for prospects
    persona_df['TotalSpent'] = persona_df['TotalSpent'].fillna(0)
    persona_df['Recency'] = persona_df['Recency'].fillna(999)
    persona_df['FavoriteCategory'] = persona_df['FavoriteCategory'].fillna("None")

    # Merge persona into final_data for easy slicing
    final_data = final_data.merge(persona_df[['ClientID', 'Persona']], on='ClientID', how='left')

    # 3. Train Models
    models_dict = {
        'w2v': train_prod2vec(final_data),
        'markov': build_markov_transition_matrices(final_data, state_col='FamilyLevel2'),
        'associations_pid': get_assoc_rules(final_data, item_col='ProductID'),
        'associations_fam': get_assoc_rules(final_data, item_col='FamilyLevel2'),
        'popular': get_global_trends(final_data), 
        'trending': get_global_trends(final_data, date=latest_date_str) 
    }
    
    return final_data, persona_df, models_dict, stocks_df

    return final_data, persona_df, models_dict, stocks_df

# --- HELPER: COUNTRY INSIGHTS ---
@st.cache_data
def build_country_insights(_fd, _persona_df, _stocks_df):
    insights = {}
    countries = _fd['ClientCountry'].unique()
    
    for cty in countries:
        cty_data = _fd[_fd['ClientCountry'] == cty]
        cty_persona = _persona_df[_persona_df['ClientCountry'] == cty] # Ensure persona_df has ClientCountry merged or we merge it here
        
        # If ClientCountry not in persona_df, merge it
        if 'ClientCountry' not in _persona_df.columns:
             # Merge from _fd just to be safe, or assume it's passed in
             # Optimization: _persona_df passed here should ideally have it.
             # We will handle it by re-merging if needed or using ClientID map
             temp_merge = _persona_df.merge(_fd[['ClientID', 'ClientCountry']].drop_duplicates('ClientID'), on='ClientID', how='left')
             cty_persona = temp_merge[temp_merge['ClientCountry'] == cty]

        # KPIs
        revenue = cty_data['SalesNetAmountEuro'].sum()
        nx_txns = len(cty_data)
        n_clients = cty_data['ClientID'].nunique()
        avg_basket = revenue / nx_txns if nx_txns > 0 else 0
        
        # Persona distribution
        pdist = cty_persona['Persona'].value_counts().to_dict()
        total_cty_cli = sum(pdist.values())

        # Top 5 Product Families
        top_fams = (
            cty_data.groupby("FamilyLevel2")["SalesNetAmountEuro"]
            .sum().nlargest(5).reset_index()
        )
        top_fams.columns = ["Product Family", "Revenue"]
        
        # Top 5 Categories
        top_cats = (
            cty_data.groupby("Category")["SalesNetAmountEuro"]
            .sum().nlargest(5).reset_index()
        )
        top_cats.columns = ["Category", "Revenue"]

        # Stock health (using StoreCountry which maps to ClientCountry usually)
        cty_stock = _stocks_df[_stocks_df['StoreCountry'] == cty]
        total_skus = cty_stock['ProductID'].nunique()
        zero_stock = int((cty_stock['Quantity'] == 0).sum())
        overstocked = int((cty_stock['Quantity'] > cty_stock['Quantity'].median() * 3).sum()) if len(cty_stock) > 0 else 0

        # Dominant persona
        dominant = max(pdist, key=pdist.get) if pdist else "Unknown"

        # Opportunities
        inactive_count = pdist.get("Inactive (Low Freq)", 0) + pdist.get("Inactive (Mid Freq)", 0)
        cold_count = pdist.get("No Transactions", 0)

        insights[cty] = {
            "revenue": revenue,
            "transactions": nx_txns,
            "clients": n_clients,
            "avg_basket": avg_basket,
            "pdist": pdist,
            "total_cty_cli": total_cty_cli,
            "top_fams": top_fams,
            "top_cats": top_cats,
            "total_skus": total_skus,
            "zero_stock": zero_stock,
            "overstocked": overstocked,
            "dominant": dominant,
            "inactive_count": inactive_count,
            "cold_count": cold_count,
        }
    return insights

# --- LOAD ---
with st.spinner('Initializing Dashboard Environment... (This may take a minute)'):
    FINAL_DATA, PERSONA_DF, MODELS, STOCKS = load_and_train_models()
    st.success("System Ready")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Filters")
countries = ["All"] + sorted(list(FINAL_DATA['ClientCountry'].dropna().unique()))
selected_country = st.sidebar.selectbox("Select Country", countries)

# Filter Data based on selection
if selected_country != "All":
    filtered_data = FINAL_DATA[FINAL_DATA['ClientCountry'] == selected_country]
else:
    filtered_data = FINAL_DATA

# Get valid clients for dropdowns
valid_clients = filtered_data['ClientID'].unique()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Executive Analytics", "Operations Intelligence", "Customer Insights"])

# ==============================================================================
# TAB 1: EXECUTIVE / MARKETING MANAGER
# ==============================================================================
with tab1:
    st.title("Executive Strategic Dashboard")
    
    # KPIs
    st.markdown("### Key Performance Indicators")
    # --- GLOBAL KPIs ---
    # Filter PERSONA_DF by country to include users with 0 transactions in counts
    sel_countries = [selected_country] if selected_country != "All" else countries[1:]
    p_filtered = PERSONA_DF[PERSONA_DF['ClientCountry'].isin(sel_countries)] if 'ClientCountry' in PERSONA_DF.columns else PERSONA_DF
    
    total_rev = filtered_data['SalesNetAmountEuro'].sum()
    total_txn = len(filtered_data)
    unique_cust = filtered_data['ClientID'].nunique()
    avg_bask = total_rev / total_txn if total_txn > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", fmt_eur(total_rev))
    k2.metric("Total Transactions", f"{total_txn:,}")
    k3.metric("Active Clients", f"{unique_cust:,}")
    k4.metric("Avg Basket Size", fmt_eur(avg_bask))

    st.markdown("---")
    
    # --- CUSTOMER SEGMENTATION FRAMEWORK ---
    st.markdown("### Customer Segmentation Framework")
    st.caption("Strategic alignment and engagement channels by customer segment.")
    
    strat_rows = []
    # Count personas for context
    counts = p_filtered.groupby('Persona')['ClientID'].nunique().to_dict()
    
    for p_name, s in PERSONA_STRATEGIES.items():
        strat_rows.append({
            "Persona": p_name,
            "Clients": counts.get(p_name, 0),
            "Strategy": s["strategy"],
            "Channel": s["channel"],
            "Priority": s["urgency"],
        })
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Category Performance Insights")
    
    st.subheader("Revenue Distribution by Segment")
    st.caption("Category revenue contribution across customer personas.")
    # Group by Category and Persona
    cat_persona = filtered_data.groupby(['FamilyLevel1', 'Persona'])['SalesNetAmountEuro'].sum().reset_index()
    # Filter for top 10 categories to avoid clutter
    top_cats = cat_persona.groupby('FamilyLevel1')['SalesNetAmountEuro'].sum().nlargest(10).index
    cat_persona = cat_persona[cat_persona['FamilyLevel1'].isin(top_cats)]
    
    fig_cp = px.bar(cat_persona, x='FamilyLevel1', y='SalesNetAmountEuro', color='Persona', 
                    title="Top 10 Categories: Revenue by Segment", barmode='stack')
    st.plotly_chart(fig_cp, use_container_width=True)
    
    # --- 1. REGIONAL STRATEGIC ANALYSIS ---
    st.markdown("### Regional Strategic Analysis")
    st.caption("Analytical insights derived from inventory levels, segment distribution, and regional trends.")
    
    # Calculate Insights
    insights = build_country_insights(filtered_data, PERSONA_DF, STOCKS)
    
    for cty, data in insights.items():
        # Helpers for badges
        top_cats = "".join(f'<span class="rec-badge">{r}</span>' for r in data['top_cats']['Category'].head(3))
        
        # Strategy Text Generation
        strat = []
        if data['inactive_count'] > 50:
            strat.append(f"<b>Win-Back Opportunity:</b> {data['inactive_count']} inactive clients found. Recommended: SMS Re-engagement.")
        if data['zero_stock'] > 5:
            strat.append(f"<b>Stock Alert:</b> {data['zero_stock']} popular items out of stock.")
        if data['dominant']:
             strat.append(f"<b>Dominant Persona:</b> {data['dominant']} (Strategy: {PERSONA_STRATEGIES.get(data['dominant'], {}).get('strategy', 'Standard')})")
            
        strat_html = "".join(f"<p style='margin:2px 0; font-size:0.9rem;'>• {s}</p>" for s in strat)
        
        with st.expander(f"Region: {cty}  |  Revenue: {fmt_eur(data['revenue'])}  |  Clients: {data['clients']}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue", fmt_eur(data['revenue']))
            c2.metric("Avg Basket", fmt_eur(data['avg_basket']))
            c3.metric("Stock Issues", f"{data['zero_stock']} SKUs")
            
            st.markdown(f"""
            <div class="country-card">
                <h3>Recommended Actions</h3>
                {strat_html}
                <br>
                <small style="color:#aaa">Top Categories: {top_cats}</small>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")

    # --- 2. ADVANCED TARGETING ENGINE ---
    st.markdown("###  Advanced Targeting Engine")
    
    t_mode = st.radio("Targeting Mode", ["Search by Product Name", "Reverse Association Rules"], horizontal=True)
    
    if t_mode == "Search by Product Name":
        search_q = st.text_input("Find likely buyers for:", placeholder='e.g. "Nike", "Running", "Gloves"')
        
        if search_q:
            # Fuzzy Filtering
            products_df = pd.read_csv('products.csv') # Optimization: Cache this or load once at top
            matches = products_df[
                products_df['FamilyLevel2'].str.contains(search_q, case=False, na=False) | 
                products_df['Category'].str.contains(search_q, case=False, na=False)
            ]
            
            if not matches.empty:
                st.success(f"Found {len(matches)} products matching '{search_q}'. Finding target clients...")
                target_families = matches['FamilyLevel2'].unique()
                
                # Find clients who bought these families BEFORE
                # Logic: If bought "Nike Shoes" before, maybe interested in new "Nike" drop?
                # Or simplistic: Find clients with High Affinity to these categories
                
                target_clients = filtered_data[filtered_data['FamilyLevel2'].isin(target_families)]['ClientID'].unique()
                
                if len(target_clients) > 0:
                    st.write(f"**{len(target_clients)} High-Propensity Clients Identified**")
                    
                    # Score them by RFM
                    targets_df = PERSONA_DF[PERSONA_DF['ClientID'].isin(target_clients)].copy()
                    
                    # Display Top 10
                    st.dataframe(targets_df[['ClientID', 'Persona', 'ClientCountry']].head(20), hide_index=True)
                else:
                    st.warning("No past buyers found for this category.")
            else:
                st.warning("No products found.")

    else:
        # EXISTING LOGIC: Reverse Associations (Keep ProductID for targeting)
        st.caption("Identify clients likely to buy a specific product based on purchase patterns.")
        
        # Get list of products ONLY where association rules exist (Consequents)
        rules = MODELS['associations_pid']
        valid_target_products = set()
        
        # Collect all products that appear on the RHS of rules
        for consequents in rules['consequents']:
            valid_target_products.update(consequents)
            
        all_products = sorted(list(valid_target_products))
        
        # Sub-mode for ID selection
        id_mode = st.radio("ID Selection", ["Select from List", "Enter Manually"], horizontal=True, key="rev_id_mode")
        
        target_product_id = None
        
        if id_mode == "Select from List":
            if not all_products:
                 st.warning("No association rules generated. Try lowering support threshold?")
            else:
                 target_product_id = st.selectbox("Select Target Product ID (Rules Available)", all_products, key="tgt_prod_full")
        else:
            txt_id = st.text_input("Enter Target Product ID", placeholder="e.g. 1053601088228117848")
            if txt_id:
                target_product_id = txt_id.strip()

        if target_product_id:
             if st.button("Find Target Audience", type="primary", key="btn_tgt_full"):
                 # Logic: Reverse Association Rules
                 rules = MODELS['associations_pid']
                 target_str = str(target_product_id)
                 relevant_rules = rules[rules['consequents'].apply(lambda x: target_str in x)]
                 
                 if not relevant_rules.empty:
                     st.write(f"Found {len(relevant_rules)} association pathways.")
                     antecedent_products = set()
                     for ante in relevant_rules['antecedents']:
                         antecedent_products.update(ante)
                     
                     st.write(f"**Key Driver Products:** {list(antecedent_products)[:5]}...")
                     
                     potential_clients = filtered_data[filtered_data['ProductID'].astype(str).isin(antecedent_products)]['ClientID'].unique()
                     existing_owners = filtered_data[filtered_data['ProductID'].astype(str) == target_str]['ClientID'].unique()
                     target_list = list(set(potential_clients) - set(existing_owners))
                     
                     st.success(f"**Identified {len(target_list)} High-Propensity Targets**")
                     
                     if target_list:
                         targets_df = PERSONA_DF[PERSONA_DF['ClientID'].isin(target_list)][['ClientID', 'Persona', 'ClientCountry']].head(10)
                         st.dataframe(targets_df, hide_index=True)
                 else:
                     st.warning("No strong association rules found leading to this product.")

    st.markdown("---")
    st.markdown("### Strategic Advisory Engine")
    st.info("Algorithmic Strategic Recommendations based on Cohort Behavior & Channel Reachability")
    
    # Strategy Logic
    persona_metrics = filtered_data.groupby('Persona').agg({
        'SalesNetAmountEuro': 'mean',
        'HasEmail': 'mean',
        'HasPhone': 'mean',
        'ClientID': 'nunique'
    }).reset_index()
    
    cols = st.columns(3)
    
    for idx, row in persona_metrics.iterrows():
        p_name = row['Persona']
        avg_val = row['SalesNetAmountEuro']
        email_reach = row['HasEmail'] * 100
        phone_reach = row['HasPhone'] * 100
        
        # Determine Strategy
        strategy_text = ""
        if "Loyal" in p_name or "Very Frequent" in p_name:
            strategy_text = "VIP Treatment: Offer exclusive early access to new collections. Focus on retention."
        elif "Inactive" in p_name:
            strategy_text = "Win-Back: Aggressive limited-time discounts for churn prevention."
        elif "No Transactions" in p_name:
            strategy_text = "Activation: Welcome series with first-purchase incentive."
        else:
            strategy_text = "Nurture: Cross-sell complementary categories to increase basket equity."
            
        # Determine Channel
        channel_text = ""
        if email_reach > 50 and phone_reach > 50:
            channel_text = f"Omnichannel: Use Email ({email_reach:.1f}%) & SMS ({phone_reach:.1f}%) for balanced impact."
        elif email_reach > 50:
            channel_text = f"Email Primary: High email opt-in ({email_reach:.1f}%). Use rich visual newsletters."
        elif phone_reach > 50:
            channel_text = f"SMS Primary: High phone opt-in ({phone_reach:.1f}%). Optimal for urgent/short alerts."
        else:
             channel_text = "Low Reach: Rely on On-Site Personalization / Retargeting Ads."

        with cols[idx % 3]:
            st.markdown(f"""
            """, unsafe_allow_html=True)

    # --- GLOBAL ANALYTICS SECTION ---
    st.markdown("---")
    st.markdown("### Geographic & Segment Analytics")
    
    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown("**Revenue by Country**")
        country_rev = filtered_data.groupby('StoreCountry')['SalesNetAmountEuro'].sum().reset_index()
        fig_cty = px.bar(
            country_rev.sort_values("SalesNetAmountEuro", ascending=True),
            x="SalesNetAmountEuro", y="StoreCountry", orientation="h",
            color="SalesNetAmountEuro", color_continuous_scale="Blues",
        )
        fig_cty.update_layout(showlegend=False, coloraxis_showscale=False, height=300, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig_cty, use_container_width=True)

    with r1b:
        st.markdown("**Persona Distribution (Current Selection)**")
        p_counts = p_filtered.groupby('Persona')['ClientID'].nunique().reset_index()
        p_counts.columns = ["Persona", "Count"]
        fig_pie = px.pie(
            p_counts, values="Count", names="Persona", hole=0.4,
            color_discrete_sequence=PALETTE,
        )
        fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown("**Top Categories by Revenue**")
        cat_rev = (
            filtered_data.groupby("Category")["SalesNetAmountEuro"]
            .sum().reset_index()
            .sort_values("SalesNetAmountEuro", ascending=False).head(12)
        )
        cat_rev.columns = ["Category", "Revenue"]
        fig_cat = px.bar(cat_rev, x="Revenue", y="Category", orientation="h",
                         color="Revenue", color_continuous_scale="Reds")
        fig_cat.update_layout(showlegend=False, coloraxis_showscale=False, height=350, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig_cat, use_container_width=True)

    with r2b:
        st.markdown("**Category x Country Heatmap**")
        top_cats = filtered_data.groupby("Category")["SalesNetAmountEuro"].sum().nlargest(10).index.tolist()
        cat_cty = (
            filtered_data[filtered_data["Category"].isin(top_cats)]
            .groupby(["StoreCountry", "Category"])["SalesNetAmountEuro"]
            .sum().reset_index()
        )
        pivot = cat_cty.pivot(index="Category", columns="StoreCountry", values="SalesNetAmountEuro").fillna(0)
        fig_hm = px.imshow(
            pivot, color_continuous_scale="Blues",
            labels=dict(x="Country", y="Category", color="Revenue"),
            aspect="auto",
        )
        fig_hm.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")
    st.markdown("**Country KPI Summary**")
    country_stats = filtered_data.groupby('StoreCountry').agg({
        'SalesNetAmountEuro': 'sum',
        'ClientID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    
    # Calculate transactions (row count per country in filtered_data)
    txn_counts = filtered_data.groupby('StoreCountry').size().reset_index(name='Transactions')
    country_stats = country_stats.merge(txn_counts, on='StoreCountry')
    
    # Avg Basket
    country_stats['Avg Basket'] = country_stats['SalesNetAmountEuro'] / country_stats['Transactions']
    
    # Formatting
    disp = country_stats.rename(columns={
        'StoreCountry': 'Country',
        'SalesNetAmountEuro': 'Revenue',
        'ClientID': 'Clients',
        'Quantity': 'Units Sold'
    })
    disp['Revenue'] = disp['Revenue'].apply(fmt_eur)
    disp['Avg Basket'] = disp['Avg Basket'].apply(fmt_eur)
    
    st.dataframe(disp[['Country', 'Revenue', 'Transactions', 'Clients', 'Units Sold', 'Avg Basket']], 
                 use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: STORE MANAGER
# ==============================================================================
with tab2:
    st.title("Operations Intelligence Hub")
    
    store_ids = sorted(filtered_data['StoreID'].dropna().unique().astype(str))
    
    if len(store_ids) > 0:
        selected_store = st.selectbox("Select Store ID", store_ids)
        
        # Filter purely for this view
        store_data = filtered_data[filtered_data['StoreID'].astype(str) == selected_store]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Operational Metrics: {selected_store}")
            st.metric("Gross Revenue", f"€{store_data['SalesNetAmountEuro'].sum():,.0f}")
            st.metric("Total Transactions", f"{len(store_data):,.0f}")
            st.metric("Average Transaction Value", f"€{store_data['SalesNetAmountEuro'].mean():,.2f}")
            
        with col2:
            st.warning("High-Velocity Inventory")
            top_products = store_data.groupby(['ProductID', 'FamilyLevel2'])['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(10)
            st.dataframe(top_products, hide_index=True)

        st.markdown("### Inventory Affinity Map")
        st.caption("Commonly associated product categories identified via historical purchasing patterns.")
        
        # Filter association rules for this store's top product FAMILIES
        top_fams = top_products['FamilyLevel2'].tolist()
        rules_df = MODELS['associations_fam']
        
        # Helper to check intersection
        def is_relevant(antecedents_frozenset):
            return any(f in antecedents_frozenset for f in top_fams)

        if not rules_df.empty:
            relevant_mask = rules_df['antecedents'].apply(is_relevant)
            relevant_rules = rules_df[relevant_mask].copy()
            
            if not relevant_rules.empty:
                # Clean up for display
                relevant_rules['antecedents'] = relevant_rules['antecedents'].apply(lambda x: list(x))
                relevant_rules['consequents'] = relevant_rules['consequents'].apply(lambda x: list(x))
                
                st.dataframe(
                    relevant_rules[['antecedents', 'consequents', 'lift', 'confidence']]
                    .sort_values('lift', ascending=False).head(10),
                    hide_index=True
                )
            else:
                st.write("No strong bundling rules found specifically for top products.")
        else:
            st.write("No association rules available.")
            
        st.markdown("---")
        st.markdown("### Inventory Optimization Monitor")
        st.caption("High-stock items identified for potential liquidation or strategic promotions.")
        
        # Get country for the selected store to filter stock
        # We can derive it from store_data if ClientCountry or StoreCountry is available
        cty_code = store_data['StoreCountry'].iloc[0] if 'StoreCountry' in store_data.columns else "FRA"
        
        cty_stock = STOCKS[STOCKS['StoreCountry'] == cty_code].copy()
        if not cty_stock.empty:
            # Merge with products to get names
            prods = pd.read_csv('products.csv')[["ProductID", "Category", "FamilyLevel1", "FamilyLevel2"]]
            # Harmonize types to avoid merge error
            cty_stock['ProductID'] = cty_stock['ProductID'].astype(str)
            prods['ProductID'] = prods['ProductID'].astype(str)
            
            cty_detail = cty_stock.merge(prods, on="ProductID", how="left")
            med_qty = cty_detail["Quantity"].median()
            over = (
                cty_detail[cty_detail["Quantity"] > med_qty]
                .sort_values("Quantity", ascending=False).head(15)
            ).rename(columns={"FamilyLevel2": "Product", "FamilyLevel1": "Sport Family"})
            
            show_cols = ["Product", "Category", "Sport Family", "Quantity"]
            st.dataframe(over[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No stock data available for this region.")
            
    else:
        st.write("No stores available for selection.")

# ==============================================================================
# TAB 3: CUSTOMER INSIGHTS
# ==============================================================================
with tab3:
    st.title("Customer Intelligence Platform")
    
    # 1. CLIENT SELECTION
    c_mode = st.radio("Selection Mode", ["Browse Samples", "Enter Client ID"], horizontal=True)
    
    selected_client_id = None
    
    if c_mode == "Browse Samples":
        # Show a few examples per persona
        sample_personas = [
            "Very Frequent (>15)", "Loyal/Top (Mid Freq)", "Loyal/Top (Low Freq)",
            "Inactive (Mid Freq)", "Inactive (Low Freq)", "No Transactions",
        ]
        samps = []
        for p in sample_personas:
            rows = PERSONA_DF[PERSONA_DF["Persona"] == p]
            if len(rows) > 0:
                samps.extend(rows.head(2)["ClientID"].tolist())
        
        if samps:
            selected_client_id = st.selectbox(
                "Select a sample client", samps,
                format_func=lambda x: f"{x}  |  {PERSONA_DF[PERSONA_DF['ClientID']==x]['Persona'].iloc[0]}"
            )
        else:
            st.warning("No clients found in personas.")
            
    else:
        txt = st.text_input("Enter Client ID", placeholder="e.g. 4508698145640552159")
        if txt:
            try:
                # Treat everything as string for robust lookup
                selected_client_id = str(txt).strip()
            except:
                selected_client_id = txt.strip()

    # 2. CLIENT DASHBOARD
    if selected_client_id is not None:
        # Robust filter: compare as strings
        client_row = PERSONA_DF[PERSONA_DF['ClientID'].astype(str) == str(selected_client_id)]
        
        if not client_row.empty:
            client = client_row.iloc[0]
            persona = client['Persona']
            country = client.get('ClientCountry', 'Unknown')
            
            # --- STRATEGY LOOKUP ---
            strat = PERSONA_STRATEGIES.get(persona, PERSONA_STRATEGIES["No Transactions"])
            
            # --- CLIENT SUMMARY CARD ---
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); border-left:5px solid #e94560; border-radius:10px; padding:20px; margin-bottom:20px;'>
                <h3 style='margin:0; color:white;'>Client {selected_client_id}</h3>
                <p style='margin:5px 0; color:#e0e0e0;'><b>Persona:</b> {persona} &nbsp;|&nbsp; <b>Country:</b> {country}</p>
                <hr style='border-color:rgba(255,255,255,0.1); margin:10px 0;'>
                <p style='margin:5px 0; font-size:0.9rem; color:#ccc;'><b>Strategy:</b> {strat['strategy']}</p>
                <p style='margin:5px 0; font-size:0.9rem; color:#ccc;'><b>Channel:</b> {strat['channel']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Spend", fmt_eur(client.get('TotalSpent', 0)))
            m2.metric("Orders", int(client.get('TransactionCount', 0)))
            m3.metric("Recency", f"{int(client.get('Recency', 0))} days")
            m4.metric("Fav Category", client.get('FavoriteCategory', 'N/A'))
            
            st.markdown("---")
            
            # --- RECOMMENDATIONS ---
            st.subheader("Predictive Recommendations")
            
            if st.button("Generate Strategic Offers", type="primary"):
                with st.spinner("Running Agents (Prod2Vec, Markov, Assoc Rules)..."):
                    try:
                        recs = get_final_recommendations(selected_client_id, persona, FINAL_DATA, MODELS, STOCKS, top_n=5)
                        
                        # Fetch Product Details
                        # optimization: lookup in a products df instead of FINAL_DATA for speed, but FINAL_DATA is loaded
                        rec_details = FINAL_DATA[FINAL_DATA['ProductID'].astype(str).isin([str(r) for r in recs])].drop_duplicates('ProductID')
                        
                        if not rec_details.empty:
                            # Create a clean display DF
                            display_recs = rec_details[['ProductID', 'FamilyLevel2', 'Category', 'PricePerUnit']].copy()
                            display_recs.columns = ['Product ID', 'Product Family', 'Category', 'Price']
                            display_recs['Price'] = display_recs['Price'].apply(fmt_eur)
                            
                            # Check Stock Logic (Visual)
                            # (Though get_final_recommendations already filters stock, being explicit helps trust)
                            display_recs['Stock Status'] = "In Stock" 
                            
                            st.dataframe(display_recs, use_container_width=True, hide_index=True)
                        else:
                            st.write(recs) # Fallback if details lookup fails
                            
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")

            # --- BASKET AFFINITY ANALYSIS ---
            last_txns = FINAL_DATA[FINAL_DATA['ClientID'] == selected_client_id].sort_values('SaleTransactionDate', ascending=False)
            if not last_txns.empty:
                last_prod = last_txns.iloc[0]['FamilyLevel2']
                
                with st.expander(f"Basket Affinity Analysis (Primary Attribution: '{last_prod}')"):
                    rules = MODELS['associations_fam']
                    relevant_rules = rules[rules['antecedents'].apply(lambda x: last_prod in x)].sort_values('lift', ascending=False).head(3)
                    
                    if not relevant_rules.empty:
                        cross_sell_data = []
                        # Get all unique consequents to lookup details
                        all_consequents = []
                        for _, row in relevant_rules.iterrows():
                            all_consequents.extend(list(row['consequents']))
                        
                        # Lookup details in FINAL_DATA
                        cross_details = FINAL_DATA[FINAL_DATA['FamilyLevel2'].isin(all_consequents)].drop_duplicates('FamilyLevel2')
                        
                        for _, detail in cross_details.iterrows():
                            # Check stock in client's country
                            stock_qty = STOCKS[(STOCKS['StoreCountry'] == country) & (STOCKS['ProductID'] == detail['ProductID'])]['Quantity'].sum()
                            cross_sell_data.append({
                                "Product": detail['FamilyLevel2'],
                                "Category": detail['Category'],
                                "Universe": detail.get('Universe', 'All'),
                                "Stock Status": f"{stock_qty:,.0f} units" if stock_qty > 0 else "Out of Stock"
                            })
                        
                        if cross_sell_data:
                            st.dataframe(pd.DataFrame(cross_sell_data), use_container_width=True, hide_index=True)
                        else:
                            st.info("No in-stock bundles found.")
                    else:
                        st.info(f"No specific bundles found for '{last_prod}'.")

        else:
            st.error("Client not found.")
