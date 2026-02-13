"""
Sports Retail Intelligence — Dashboard
========================================
AI-powered product recommendation engine for personalised marketing campaigns.
Loads pre-computed artifacts from processed/ folder.

Before first use:  python precompute.py
Then run:          streamlit run dashboard.py
"""

import os, json, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")
from Final_Orchestra_2 import get_final_recommendations

PROCESSED = "processed"

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Next Purchase — Sports Retail Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #e94560;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #ccc; }
    [data-testid="stMetricValue"] { font-size: 1.30rem; font-weight: 700; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        overflow-x: auto;
        flex-wrap: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.85rem;
        white-space: nowrap;
        min-width: fit-content;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #e94560 !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="false"] {
        color: #aaa !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border-left: 3px solid #e94560;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff; }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #aaa; }
    .streamlit-expanderHeader { font-size: 1rem; font-weight: 600; }
    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Tab container: no clipping */
    .stTabs {
        margin-top: 0.5rem;
        overflow: visible !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        padding-top: 4px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        overflow: visible !important;
    }
    /* Ensure all text visible on dark theme */
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, li, span, label, .stMarkdown { color: #e0e0e0; }
    .stCaption, .stCaption p { color: #aaa !important; }
    .stSelectbox label, .stTextInput label, .stRadio label { color: #ccc !important; }
    .stSpinner > div { color: #e0e0e0 !important; }
    /* Loading spinner visibility */
    .stSpinner { background: rgba(0,0,0,0.6); border-radius: 8px; padding: 12px; }
    /* Country recommendation cards */
    .country-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .country-card h3 { margin: 0 0 8px 0; color: #e94560 !important; font-size: 1.1rem; }
    .country-card p { color: #e0e0e0; margin: 4px 0; font-size: 0.88rem; }
    .country-card .kpi { color: #ffffff; font-weight: 700; }
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

# ─── Constants ────────────────────────────────────────────────────────
PALETTE = [
    "#1a1a2e", "#16213e", "#0f3460", "#e94560",
    "#533483", "#2b9348", "#d62828", "#f77f00",
]

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


# ─── Helpers ──────────────────────────────────────────────────────────
def fmt_eur(val):
    return f"\u20ac{val:,.0f}" if abs(val) >= 1000 else f"\u20ac{val:,.2f}"


def gender_label(g):
    return {"M": "Men", "F": "Women"}.get(g, "All")


# ═══════════════════════════════════════════════════════════════════════
# FAST LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading data...")
def load_precomputed():
    final_data = pd.read_csv(os.path.join(PROCESSED, "final_data.csv.gz"))
    persona_df = pd.read_csv(os.path.join(PROCESSED, "persona.csv.gz"))
    client_features = pd.read_csv(os.path.join(PROCESSED, "client_features.csv.gz"))
    associations_df = pd.read_csv(os.path.join(PROCESSED, "associations.csv.gz"))
    popular_df = pd.read_csv(os.path.join(PROCESSED, "popular.csv.gz"))
    trending_df = pd.read_csv(os.path.join(PROCESSED, "trending.csv.gz"))
    affinity_df = pd.read_csv(os.path.join(PROCESSED, "affinity.csv.gz"))
    country_kpis = pd.read_csv(os.path.join(PROCESSED, "country_kpis.csv.gz"))
    ctrend_path = os.path.join(PROCESSED, "country_trending.csv.gz")
    country_trending = pd.read_csv(ctrend_path) if os.path.exists(ctrend_path) else pd.DataFrame()
    with open(os.path.join(PROCESSED, "metadata.json")) as f:
        metadata = json.load(f)
    return (final_data, persona_df, client_features, associations_df,
            popular_df, trending_df, affinity_df, country_kpis,
            country_trending, metadata)


@st.cache_resource(show_spinner="Loading Prod2Vec model...")
def load_w2v():
    return Word2Vec.load(os.path.join(PROCESSED, "w2v.model"))


@st.cache_data(show_spinner="Loading Markov matrices...")
def load_markov():
    return joblib.load(os.path.join(PROCESSED, "markov.pkl"))


@st.cache_data(show_spinner="Loading catalogues...")
def load_catalogues():
    return pd.read_csv("products.csv"), pd.read_csv("stocks.csv")


@st.cache_data
def build_product_lookup(_products_df):
    lookup = {}
    for _, r in _products_df.iterrows():
        lookup[r["ProductID"]] = {
            "Product": r.get("FamilyLevel2", "N/A"),
            "Category": r.get("Category", "N/A"),
            "Sport Family": r.get("FamilyLevel1", "N/A"),
            "Gender": r.get("Universe", "N/A"),
        }
    return lookup


@st.cache_data
def build_stock_index(_stocks_df):
    idx = {}
    for _, r in _stocks_df.iterrows():
        idx[(r["StoreCountry"], r["ProductID"])] = int(r["Quantity"])
    return idx


@st.cache_data
def build_reverse_product_index(_final_data):
    """product-family -> set of ClientIDs who purchased it."""
    fam_clients = (
        _final_data.groupby("FamilyLevel2")["ClientID"]
        .apply(lambda x: set(x.unique()))
        .to_dict()
    )
    cat_clients = (
        _final_data.groupby("Category")["ClientID"]
        .apply(lambda x: set(x.unique()))
        .to_dict()
    )
    return fam_clients, cat_clients


@st.cache_data
def get_persona_map(_persona_df):
    return dict(zip(_persona_df["ClientID"], _persona_df["Persona"]))


# ─── Check processed/ exists ─────────────────────────────────────────
if not os.path.isdir(PROCESSED) or not os.path.exists(os.path.join(PROCESSED, "metadata.json")):
    st.error("**Pre-computed data not found.** Run `python precompute.py` first.")
    st.stop()

# ─── Load everything ─────────────────────────────────────────────────
try:
    (final_data, persona_df, client_features, associations_df,
     popular_df, trending_df, affinity_df, country_kpis_df,
     country_trending_df, metadata) = load_precomputed()
    w2v_model = load_w2v()
    markov_model = load_markov()
    products_df, stocks_df = load_catalogues()
    product_lookup = build_product_lookup(products_df)
    stock_index = build_stock_index(stocks_df)
    fam_clients_idx, cat_clients_idx = build_reverse_product_index(final_data)
    persona_map = get_persona_map(persona_df)
    models_dict = {
        "w2v": w2v_model, "markov": markov_model,
        "associations": associations_df, "popular": popular_df, "trending": trending_df,
    }
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# ─── Derived values ──────────────────────────────────────────────────
ALL_COUNTRIES = metadata.get("countries",
    sorted(final_data["StoreCountry"].dropna().unique().tolist()))
n_clients = metadata.get("n_clients_total", client_features["ClientID"].nunique())
n_products = metadata.get("n_products", products_df["ProductID"].nunique())
n_countries = metadata.get("n_countries", len(ALL_COUNTRIES))

# ─── Cached recommendation wrapper ───────────────────────────────────
_data_hash = metadata.get("pipeline_time_seconds", 0)

@st.cache_data(show_spinner=False, ttl=600)
def cached_recs(cid, persona, _hash, top_n=10):
    return get_final_recommendations(cid, persona, final_data,
                                     models_dict, stocks_df, top_n=top_n)


def enrich_recs(rec_ids, country):
    rows = []
    for rank, pid in enumerate(rec_ids, 1):
        info = product_lookup.get(pid, {
            "Product": "Unknown", "Category": "N/A",
            "Sport Family": "N/A", "Gender": "N/A",
        })
        stock_qty = stock_index.get((country, pid), 0)
        rows.append({"Rank": rank, **info, "Stock": stock_qty, "ProductID": pid})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### The Next Purchase")
    st.caption("AI-powered product recommendation engine")
    st.markdown("---")
    st.metric("Clients", f"{n_clients:,}")
    st.metric("Products", f"{n_products:,}")
    st.metric("Countries", n_countries)
    st.markdown("---")



# ═══════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F3AF  Recommendations",
    "\U0001F50D  Product Targeting",
    "\U0001F4CA  Strategy",
    "\U0001F4E6  Inventory",
])


# =====================================================================
# TAB 1 — CLIENT → PRODUCT RECOMMENDATIONS
# =====================================================================
with tab1:
    st.markdown("#### What should we recommend to this client?")
    st.caption(
        "Enter a Client ID to get personalised, stock-aware product recommendations. "
        "The AI engine blends Prod2Vec, Markov chains, association rules, and popularity "
        "models — then filters by available stock in the client's country."
    )

    c_mode = st.radio("Find Client", ["Enter Client ID", "Browse Samples"],
                      horizontal=True, key="t1_mode")

    selected_cid = None
    if c_mode == "Browse Samples":
        sample_personas = [
            "Very Frequent (>15)", "Loyal/Top (Mid Freq)", "Loyal/Top (Low Freq)",
            "Inactive (Mid Freq)", "Inactive (Low Freq)", "No Transactions",
        ]
        samps = []
        for p in sample_personas:
            rows = client_features[client_features["Persona"] == p]
            if len(rows) > 0:
                samps.extend(rows.head(2)["ClientID"].tolist())
        selected_cid = st.selectbox(
            "Select a sample client", samps, key="t1_sel",
            format_func=lambda x: f"{x}  |  {persona_map.get(x, 'Unknown')}",
        )
    else:
        txt = st.text_input("Client ID", key="t1_id",
                            placeholder="e.g. 4508698145640552159")
        if txt:
            try:
                selected_cid = int(txt)
            except ValueError:
                st.warning("Enter a numeric Client ID.")

    if selected_cid is not None:
        cf_row = client_features[client_features["ClientID"] == selected_cid]
        if len(cf_row) == 0:
            p_row = persona_df[persona_df["ClientID"] == selected_cid]
            if len(p_row) == 0:
                st.warning(f"Client {selected_cid} not found.")
                st.stop()
            client = p_row.iloc[0]
            is_cold = True
            persona_name = "No Transactions"
        else:
            client = cf_row.iloc[0]
            is_cold = client.get("Persona", "") == "No Transactions"
            persona_name = client.get("Persona", "Unknown")

        country = client.get("ClientCountry", client.get("StoreCountry", "N/A"))
        strat = PERSONA_STRATEGIES.get(persona_name, PERSONA_STRATEGIES["No Transactions"])

        # Client summary card
        st.markdown("---")
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05); "
            f"border-left:5px solid #e94560; border-radius:10px; padding:20px 28px; "
            f"margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.25);'>"
            f"<h3 style='margin:0; color:#ffffff;'>Client {selected_cid}</h3>"
            f"<p style='margin:4px 0 0; color:#e0e0e0; font-size:0.92rem;'>"
            f"Persona: <b>{persona_name}</b> &nbsp;|&nbsp; Country: <b>{country}</b></p>"
            f"<p style='margin:4px 0 0; color:#aaa; font-size:0.85rem;'>"
            f"Strategy: {strat['strategy']} &nbsp;|&nbsp; "
            f"Channel: {strat['channel']}</p></div>",
            unsafe_allow_html=True,
        )

        # Client KPIs
        if not is_cold:
            mc = st.columns(6)
            mc[0].metric("Total Spend", fmt_eur(float(client.get("TotalSpent", 0))))
            mc[1].metric("Transactions", int(client.get("TransactionCount", 0)))
            mc[2].metric("Avg Order", fmt_eur(float(client.get("AvgOrderValue", 0))))
            mc[3].metric("Recency", f"{int(client.get('Recency', 0))}d")
            mc[4].metric("RFM Score", int(client.get("RFM_Score", 0)))
            mc[5].metric("Fav Category", client.get("FavoriteCategory", "N/A"))
        else:
            mc = st.columns(3)
            mc[0].metric("Gender", gender_label(client.get("ClientGender", "U")))
            mc[1].metric("Country", str(country))
            mc[2].metric("Segment", client.get("ClientSegment", "N/A"))

        # AI Recommendations
        st.markdown("---")
        st.markdown("**AI-Generated Product Recommendations**")
        st.caption(
            f"Considering available stock in **{country}** | "
            f"Models: Prod2Vec + Markov + Association Rules + Popularity"
        )

        with st.spinner("Generating recommendations..."):
            try:
                recs = cached_recs(selected_cid, persona_name, _data_hash, top_n=10)
                if recs:
                    rec_df = enrich_recs(recs, country)
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No recommendations — stock may be empty for this region.")
            except Exception as exc:
                st.warning(f"Recommendation engine error: {exc}")

        # Cross-sell expander
        if not is_cold:
            user_txns = final_data[final_data["ClientID"] == selected_cid]
            if len(user_txns) > 0:
                last_fam = user_txns.iloc[-1].get("FamilyLevel2", None)
                if last_fam:
                    assoc_row = associations_df[associations_df["FamilyLevel2"] == last_fam]
                    if len(assoc_row) > 0:
                        with st.expander(
                            f"Cross-sell: Clients who bought {last_fam} also buy...",
                            expanded=False,
                        ):
                            cty_avail = set(
                                stocks_df[
                                    (stocks_df["StoreCountry"] == country)
                                    & (stocks_df["Quantity"] > 0)
                                ]["ProductID"]
                            )
                            cross_prods = []
                            for _, arow in assoc_row.iterrows():
                                for col in ["Reco1", "Reco2", "Reco3"]:
                                    rf = arow.get(col)
                                    if pd.isna(rf):
                                        continue
                                    fp_df = products_df[
                                        (products_df["FamilyLevel2"] == rf)
                                        & (products_df["ProductID"].isin(cty_avail))
                                    ]
                                    for _, fp in fp_df.head(3).iterrows():
                                        qty = stock_index.get((country, fp["ProductID"]), 0)
                                        cross_prods.append({
                                            "Product": fp.get("FamilyLevel2", "N/A"),
                                            "Category": fp.get("Category", "N/A"),
                                            "Gender": fp.get("Universe", "N/A"),
                                            "Stock": qty,
                                        })
                            if cross_prods:
                                st.dataframe(pd.DataFrame(cross_prods),
                                             use_container_width=True, hide_index=True)
                            else:
                                st.info(f"No in-stock cross-sell for '{last_fam}' in {country}.")


# =====================================================================
# TAB 2 — PRODUCT → CLIENT TARGETING  (reverse lookup)
# =====================================================================
with tab2:
    st.markdown("#### Who should we recommend this product to?")
    st.caption(
        "Search for a product, then see which clients are the best targets "
        "based on their purchase history, persona, and country stock availability."
    )

    search_q = st.text_input(
        "Search by product name, category, or Product ID",
        key="p2c_search",
        placeholder='e.g. "Nike Dri-FIT", "Football", or a Product ID',
    )

    if search_q:
        try:
            pid_q = int(search_q)
            matches = products_df[products_df["ProductID"] == pid_q]
        except ValueError:
            matches = products_df[
                products_df["FamilyLevel2"].str.contains(search_q, case=False, na=False)
                | products_df["Category"].str.contains(search_q, case=False, na=False)
                | products_df["FamilyLevel1"].str.contains(search_q, case=False, na=False)
            ]

        if len(matches) == 0:
            st.info("No products found matching your search.")
        else:
            st.caption(f"Found **{len(matches)}** matching products.")

            display_matches = matches.head(20).rename(columns={
                "FamilyLevel2": "Product", "FamilyLevel1": "Sport Family", "Universe": "Gender",
            })
            st.dataframe(
                display_matches[["Product", "Category", "Sport Family", "Gender", "ProductID"]],
                use_container_width=True, hide_index=True,
            )

            # Stock availability
            matched_ids = matches["ProductID"].tolist()
            stock_info = stocks_df[stocks_df["ProductID"].isin(matched_ids)]
            if len(stock_info) > 0:
                st.markdown("**Stock Availability by Country**")
                stock_pivot = stock_info.pivot_table(
                    index="ProductID", columns="StoreCountry",
                    values="Quantity", aggfunc="sum",
                ).fillna(0).astype(int)
                stock_pivot = stock_pivot.reset_index().merge(
                    products_df[["ProductID", "FamilyLevel2"]].rename(
                        columns={"FamilyLevel2": "Product"}
                    ), on="ProductID", how="left",
                )
                cols = ["Product"] + [c for c in stock_pivot.columns
                                      if c not in ["Product", "ProductID"]]
                st.dataframe(stock_pivot[cols], use_container_width=True, hide_index=True)

            # Find target clients
            st.markdown("---")
            st.markdown("**Recommended Target Clients**")

            sel_country = st.selectbox(
                "Filter by country (stock-aware targeting)",
                ["All"] + ALL_COUNTRIES, key="p2c_country",
            )

            matched_families = matches["FamilyLevel2"].unique().tolist()
            matched_categories = matches["Category"].unique().tolist()

            target_cids = set()
            for fam in matched_families:
                target_cids |= fam_clients_idx.get(fam, set())
            for cat in matched_categories:
                target_cids |= cat_clients_idx.get(cat, set())

            if target_cids:
                target_df = client_features[client_features["ClientID"].isin(target_cids)].copy()

                if sel_country != "All":
                    cc = "ClientCountry" if "ClientCountry" in target_df.columns else "StoreCountry"
                    target_df = target_df[target_df[cc] == sel_country]

                if len(target_df) > 0:
                    # Target score: higher RFM + more txns + lower recency
                    target_df["TargetScore"] = (
                        target_df["RFM_Score"].fillna(0) * 2
                        + target_df["TransactionCount"].fillna(0)
                        - target_df["Recency"].fillna(999) / 100
                    )
                    target_df = target_df.sort_values("TargetScore", ascending=False)

                    cols_map = {
                        "ClientID": "Client ID", "Persona": "Persona",
                        "TransactionCount": "Transactions",
                        "TotalSpent": "Total Spend",
                        "Recency": "Recency (days)",
                        "RFM_Score": "RFM Score",
                        "FavoriteCategory": "Fav Category",
                    }
                    avail = {k: v for k, v in cols_map.items() if k in target_df.columns}
                    show_df = target_df[list(avail.keys())].head(50).rename(columns=avail)
                    if "Total Spend" in show_df.columns:
                        show_df["Total Spend"] = show_df["Total Spend"].apply(fmt_eur)

                    st.caption(
                        f"**{len(target_df):,} potential clients** have purchased "
                        f"from these product families / categories. Showing top 50:"
                    )
                    st.dataframe(show_df, use_container_width=True, hide_index=True)

                    # Persona breakdown chart
                    st.markdown("**Target Audience by Persona**")
                    pdist = target_df["Persona"].value_counts().reset_index()
                    pdist.columns = ["Persona", "Clients"]
                    fig = px.bar(
                        pdist, x="Clients", y="Persona", orientation="h",
                        color="Persona", color_discrete_sequence=PALETTE,
                    )
                    fig.update_layout(height=260, showlegend=False,
                                      margin=dict(l=0, r=10, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No matching clients in {sel_country}.")
            else:
                st.info("No purchase history found for these product families.")


# =====================================================================
# TAB 3 — STRATEGIC OVERVIEW
# =====================================================================
with tab3:
    st.markdown("#### Executive Marketing Strategy")
    st.caption("Country-wise marketing recommendations for the executive team — powered by AI persona analysis, purchase trends, and stock data.")

    # Top-level KPIs
    total_rev = country_kpis_df["total_revenue"].sum()
    total_txn = country_kpis_df["n_transactions"].sum()
    total_cli = country_kpis_df["n_clients"].sum()
    avg_bask = total_rev / total_txn if total_txn else 0

    k = st.columns(4)
    k[0].metric("Total Revenue", fmt_eur(total_rev))
    k[1].metric("Total Transactions", f"{total_txn:,}")
    k[2].metric("Active Clients", f"{total_cli:,}")
    k[3].metric("Avg Basket Size", fmt_eur(avg_bask))

    st.markdown("---")

    # Persona Strategy Matrix
    st.markdown("**Persona Strategy Matrix**")
    st.caption("How the AI recommendation engine adapts to each customer segment.")
    strat_rows = []
    for p_name, s in PERSONA_STRATEGIES.items():
        count = int((persona_df["Persona"] == p_name).sum())
        strat_rows.append({
            "Persona": p_name,
            "Clients": f"{count:,}",
            "Strategy": s["strategy"],
            "Channel": s["channel"],
            "Priority": s["urgency"],
        })
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── COUNTRY-WISE MARKETING RECOMMENDATIONS ──────────────────
    st.markdown("### Country-by-Country Marketing Playbook")
    st.caption("For each market: audience breakdown, top-performing categories, stock readiness, and actionable recommendations.")

    # Build per-country insights (cached)
    @st.cache_data
    def build_country_insights(_fd, _persona_df, _stocks_df, _country_kpis):
        insights = {}
        for _, crow in _country_kpis.iterrows():
            cty = crow["StoreCountry"]
            cty_data = _fd[_fd["StoreCountry"] == cty]
            cty_persona = _persona_df[_persona_df["ClientCountry"] == cty]

            # Persona distribution for this country
            pdist = cty_persona["Persona"].value_counts().to_dict()
            total_cty_cli = sum(pdist.values())

            # Top 5 categories by revenue
            top_cats = (
                cty_data.groupby("Category")["SalesNetAmountEuro"]
                .sum().nlargest(5).reset_index()
            )
            top_cats.columns = ["Category", "Revenue"]

            # Top 5 product families by revenue
            top_fams = (
                cty_data.groupby("FamilyLevel2")["SalesNetAmountEuro"]
                .sum().nlargest(5).reset_index()
            )
            top_fams.columns = ["Product Family", "Revenue"]

            # Stock health
            cty_stock = _stocks_df[_stocks_df["StoreCountry"] == cty]
            total_skus = cty_stock["ProductID"].nunique()
            zero_stock = int((cty_stock["Quantity"] == 0).sum())
            overstocked = int((cty_stock["Quantity"] > cty_stock["Quantity"].median() * 3).sum()) if len(cty_stock) > 0 else 0

            # Dominant persona
            dominant = max(pdist, key=pdist.get) if pdist else "Unknown"

            # Win-back opportunity = inactive clients
            inactive_count = pdist.get("Inactive (Low Freq)", 0) + pdist.get("Inactive (Mid Freq)", 0)
            cold_count = pdist.get("No Transactions", 0)

            insights[cty] = {
                "revenue": crow["total_revenue"],
                "transactions": int(crow["n_transactions"]),
                "clients": int(crow["n_clients"]),
                "avg_basket": crow["avg_basket"],
                "pdist": pdist,
                "total_cty_cli": total_cty_cli,
                "top_cats": top_cats,
                "top_fams": top_fams,
                "total_skus": total_skus,
                "zero_stock": zero_stock,
                "overstocked": overstocked,
                "dominant": dominant,
                "inactive_count": inactive_count,
                "cold_count": cold_count,
            }
        return insights

    country_insights = build_country_insights(final_data, persona_df, stocks_df, country_kpis_df)

    COUNTRY_NAMES = {
        "FRA": "France", "GBR": "United Kingdom", "DEU": "Germany",
        "USA": "United States", "BRA": "Brazil", "AUS": "Australia", "ARE": "UAE",
    }

    for cty in sorted(country_insights.keys()):
        ins = country_insights[cty]
        cty_name = COUNTRY_NAMES.get(cty, cty)

        # Build recommendation bullets based on the data
        recs = []
        strat_info = PERSONA_STRATEGIES.get(ins["dominant"], {})
        recs.append(f"Primary audience is <b>{ins['dominant']}</b> — {strat_info.get('strategy', 'personalised offers')}")

        if ins["inactive_count"] > 0:
            pct = ins["inactive_count"] / ins["total_cty_cli"] * 100 if ins["total_cty_cli"] else 0
            recs.append(f"Win-back opportunity: <b>{ins['inactive_count']:,}</b> inactive clients ({pct:.1f}%) — run re-engagement campaign via Email + SMS")

        if ins["cold_count"] > 100:
            recs.append(f"Activation potential: <b>{ins['cold_count']:,}</b> registered but never purchased — welcome offer on trending products")

        if ins["zero_stock"] > 0:
            pct_zero = ins["zero_stock"] / ins["total_skus"] * 100 if ins["total_skus"] else 0
            recs.append(f"Stock alert: <b>{ins['zero_stock']}</b> SKUs at zero stock ({pct_zero:.0f}%) — replenish before campaign launch")

        if ins["overstocked"] > 0:
            recs.append(f"Promotion opportunity: <b>{ins['overstocked']}</b> overstocked SKUs — push via discount campaigns to clear inventory")

        top_cat_names = ins["top_cats"]["Category"].tolist()
        top_fam_names = ins["top_fams"]["Product Family"].tolist()

        cat_badges = "".join(f'<span class="rec-badge">{c}</span>' for c in top_cat_names)
        fam_badges = "".join(f'<span class="rec-badge">{f}</span>' for f in top_fam_names)
        rec_html = "".join(f"<p style='margin:2px 0; color:#e0e0e0; font-size:0.85rem;'>• {r}</p>" for r in recs)

        with st.expander(f"{cty_name} ({cty}) — {fmt_eur(ins['revenue'])} revenue, {ins['clients']:,} clients", expanded=False):
            mc = st.columns(4)
            mc[0].metric("Revenue", fmt_eur(ins["revenue"]))
            mc[1].metric("Transactions", f"{ins['transactions']:,}")
            mc[2].metric("Clients", f"{ins['clients']:,}")
            mc[3].metric("Avg Basket", fmt_eur(ins["avg_basket"]))

            st.markdown(
                f"<div class='country-card'>"
                f"<h3>Marketing Recommendations</h3>"
                f"{rec_html}"
                f"<p style='margin:10px 0 4px; color:#aaa; font-size:0.82rem;'><b>Top Categories:</b></p>{cat_badges}"
                f"<p style='margin:10px 0 4px; color:#aaa; font-size:0.82rem;'><b>Top Products:</b></p>{fam_badges}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Persona breakdown chart for this country
            p_data = pd.DataFrame([
                {"Persona": p, "Clients": c} for p, c in ins["pdist"].items()
            ]).sort_values("Clients", ascending=True)
            if len(p_data) > 0:
                fig = px.bar(
                    p_data, x="Clients", y="Persona", orientation="h",
                    color="Persona", color_discrete_sequence=PALETTE,
                )
                fig.update_layout(
                    height=220, showlegend=False,
                    margin=dict(l=0, r=10, t=5, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── GLOBAL CHARTS ───────────────────────────────────────────
    st.markdown("### Global Analytics")

    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown("**Revenue by Country**")
        fig = px.bar(
            country_kpis_df.sort_values("total_revenue", ascending=True),
            x="total_revenue", y="StoreCountry", orientation="h",
            color="total_revenue", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=300, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        st.markdown("**Persona Distribution (Global)**")
        persona_counts = persona_df["Persona"].value_counts().reset_index()
        persona_counts.columns = ["Persona", "Count"]
        fig = px.pie(
            persona_counts, values="Count", names="Persona", hole=0.4,
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown("**Top Categories by Revenue**")
        cat_rev = (
            final_data.groupby("Category")["SalesNetAmountEuro"]
            .sum().reset_index()
            .sort_values("SalesNetAmountEuro", ascending=False).head(12)
        )
        cat_rev.columns = ["Category", "Revenue"]
        fig = px.bar(cat_rev, x="Revenue", y="Category", orientation="h",
                     color="Revenue", color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=350, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        st.markdown("**Category x Country Heatmap**")
        top_cats = (
            final_data.groupby("Category")["SalesNetAmountEuro"]
            .sum().nlargest(10).index.tolist()
        )
        cat_cty = (
            final_data[final_data["Category"].isin(top_cats)]
            .groupby(["StoreCountry", "Category"])["SalesNetAmountEuro"]
            .sum().reset_index()
        )
        pivot = cat_cty.pivot(
            index="Category", columns="StoreCountry", values="SalesNetAmountEuro"
        ).fillna(0)
        fig = px.imshow(
            pivot, color_continuous_scale="Blues",
            labels=dict(x="Country", y="Category", color="Revenue"),
            aspect="auto",
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Country KPI Summary**")
    disp = country_kpis_df.copy()
    disp.columns = [
        "Country", "Revenue", "Transactions", "Clients",
        "Products", "Avg Basket", "Median Basket",
    ]
    disp["Revenue"] = disp["Revenue"].apply(fmt_eur)
    disp["Avg Basket"] = disp["Avg Basket"].apply(fmt_eur)
    disp["Median Basket"] = disp["Median Basket"].apply(fmt_eur)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# =====================================================================
# TAB 4 — INVENTORY & BUNDLING
# =====================================================================
with tab4:
    st.markdown("#### Inventory & Bundling")
    sel_cty = st.selectbox(
        "Select Country", ALL_COUNTRIES,
        index=ALL_COUNTRIES.index("FRA") if "FRA" in ALL_COUNTRIES else 0,
        key="t4_country",
    )
    st.markdown("---")

    inv_a, inv_b = st.tabs(["Trending & Bundles", "Stock Health"])

    with inv_a:
        st.markdown("**Trending Products — Highest Demand**")
        if len(country_trending_df) > 0:
            cty_hero = country_trending_df[country_trending_df["StoreCountry"] == sel_cty]
            if len(cty_hero) > 0:
                hero = cty_hero.head(15).copy()
                if "ProductID" in hero.columns:
                    hero = hero.merge(
                        products_df[["ProductID", "Category", "FamilyLevel1",
                                     "FamilyLevel2", "Universe"]],
                        on="ProductID", how="left",
                    ).rename(columns={
                        "FamilyLevel2": "Product", "FamilyLevel1": "Sport Family",
                        "Universe": "Gender",
                    })
                    score_cols = [c for c in hero.columns
                                  if c not in ["StoreCountry", "Source", "ProductID",
                                                "Category", "Sport Family", "Product", "Gender"]]
                    display_order = ["Product", "Category", "Sport Family", "Gender"] + score_cols
                    display_order = [c for c in display_order if c in hero.columns]
                    st.dataframe(hero[display_order], use_container_width=True, hide_index=True)
                else:
                    st.dataframe(hero, use_container_width=True, hide_index=True)
            else:
                st.info(f"No trending data for {sel_cty}.")
        else:
            st.dataframe(trending_df.head(15), use_container_width=True, hide_index=True)

        st.markdown("---")

        st.markdown("**Bundle Recommendations — Products to Sell Together**")
        st.caption("Based on co-purchase analysis of product families.")
        if len(associations_df) > 0:
            for i, (_, row) in enumerate(associations_df.head(8).iterrows()):
                recos = [r for r in [row.get("Reco1"), row.get("Reco2"), row.get("Reco3")]
                         if pd.notna(r)]
                if recos:
                    with st.expander(
                        f"{row['FamilyLevel2']} + {', '.join(recos)}",
                        expanded=(i == 0),
                    ):
                        cty_avail = set(
                            stocks_df[
                                (stocks_df["StoreCountry"] == sel_cty)
                                & (stocks_df["Quantity"] > 0)
                            ]["ProductID"]
                        )
                        items = []
                        for rf in recos:
                            fp_df = products_df[
                                (products_df["FamilyLevel2"] == rf)
                                & (products_df["ProductID"].isin(cty_avail))
                            ].head(3)
                            for _, fp in fp_df.iterrows():
                                qty = stock_index.get((sel_cty, fp["ProductID"]), 0)
                                items.append({
                                    "Product": fp.get("FamilyLevel2", "N/A"),
                                    "Category": fp.get("Category", "N/A"),
                                    "Gender": fp.get("Universe", "N/A"),
                                    "Stock": qty,
                                })
                        if items:
                            st.dataframe(pd.DataFrame(items),
                                         use_container_width=True, hide_index=True)
                        else:
                            st.info(f"No in-stock bundle products for {sel_cty}.")

        st.markdown("---")

        st.markdown("**Promotion Candidates — Overstocked Products**")
        cty_stock = stocks_df[stocks_df["StoreCountry"] == sel_cty].copy()
        if len(cty_stock) > 0:
            cty_detail = cty_stock.merge(
                products_df[["ProductID", "Category", "FamilyLevel1", "FamilyLevel2"]],
                on="ProductID", how="left",
            )
            med_qty = cty_detail["Quantity"].median()
            over = (
                cty_detail[cty_detail["Quantity"] > med_qty]
                .sort_values("Quantity", ascending=False).head(20)
            ).rename(columns={"FamilyLevel2": "Product", "FamilyLevel1": "Sport Family"})
            show = [c for c in ["Product", "Category", "Sport Family", "Quantity"]
                    if c in over.columns]
            if len(over) > 0:
                st.dataframe(over[show], use_container_width=True, hide_index=True)

    with inv_b:
        st.markdown("**Stock Overview by Country**")
        stk_agg = stocks_df.groupby("StoreCountry").agg(
            products=("ProductID", "nunique"),
            total_units=("Quantity", "sum"),
            zero_stock=("Quantity", lambda x: (x == 0).sum()),
        ).reset_index()
        stk_agg["zero_pct"] = (stk_agg["zero_stock"] / stk_agg["products"] * 100).round(1)
        stk_agg.columns = [
            "Country", "Products", "Total Units", "Zero-Stock Items", "Zero-Stock %",
        ]
        st.dataframe(stk_agg, use_container_width=True, hide_index=True)

        fig = px.bar(
            stk_agg, x="Country", y="Total Units",
            color="Zero-Stock %", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

