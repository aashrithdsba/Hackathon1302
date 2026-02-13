# STRIDE: AI-Powered Retail Analytics Dashboard

STRIDE is a next-generation customer intelligence platform designed for global retail brands. It leverages advanced machine learning to transform raw transaction data into actionable marketing playbooks, store-level bundling strategies, and hyper-personalized customer recommendations.

## 🚀 Key Features

*   **Executive Marketing Playbook**: Country-specific strategy generation, stock-health alerts, and win-back opportunities for inactive segments.
*   **Targeting Engine**: Search-by-product fuzzy matching to identify high-propensity customer cohorts.
*   **Customer Personalization**: 
    *   **Prod2Vec & Markov Models**: Predicting the "Next Likely Purchase" based on session sequences.
    *   **RFM Segmentation**: Automated persona classification (VIP, At-Risk, Loyal, etc.).
    *   **Cross-Sell Expander**: Association-rule-based bundling suggestions (e.g., "People who bought X also bought Y").
*   **Store Manager Toolkit**: Localized performance metrics and in-store bundling opportunities.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (with custom Dark Mode CSS)
*   **Analytics**: Python (Pandas, NumPy)
*   **Visualization**: Plotly Express & Graph Objects
*   **Machine Learning**: 
    *   `Gensim` (Word2Vec/Prod2Vec)
    *   Custom Markov Chain Transition Matrices
    *   Association Rule Mining (Market Basket Analysis)

## 📂 Project Structure

*   `app.py`: The main Streamlit application and UI orchestrator.
*   `Association.py`: Custom logic for multi-level (Product & Family) association rules.
*   `Data_cleaning_FE.py`: Robust feature engineering pipeline (RFM, Seasonality, Global Events).
*   `persona.py`: Logic for building customer persona profiles.
*   `sequence_models.py`: Implementations for Prod2Vec and Markov Chain predictions.
*   `Final_Orchestra_2.py`: Integration layer for the recommendation engine.

## 🏃 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas plotly gensim joblib
    ```

2.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---
*Created for the 2026 Hackathon - Team STRIDE*
