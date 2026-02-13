
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import warnings
from collections import defaultdict

# =============================================================================
# 1. PROD2VEC (Product Embeddings)
# =============================================================================

def train_prod2vec(df, vector_size=32, window=3, min_count=2, epochs=10):
    """
    Trains a Word2Vec model on product purchase sequences.
    
    Args:
        df (pd.DataFrame): Transaction data containing 'ClientID', 'SaleTransactionDate', 'ProductID'.
        vector_size (int): Dimension of the embedding vectors.
        window (int): Maximum distance between current and predicted product within a sentence.
        min_count (int): Ignores all products with total frequency lower than this.
        epochs (int): Number of iterations (epochs) over the corpus.
        
    Returns:
        gensim.models.Word2Vec: The trained model.
    """
    print(f"🔄 Training Prod2Vec model (Vector Size: {vector_size})...")
    
    # Ensure sorted by date for correct sequence
    df_sorted = df.sort_values(['ClientID', 'SaleTransactionDate'])
    
    # Group by Client to form "sentences" (sequences of product IDs)
    # We convert ProductID to string as Gensim expects list of strings
    sequences = df_sorted.groupby('ClientID')['ProductID'].apply(
        lambda x: [str(pid) for pid in x]
    ).tolist()
    
    # Train Word2Vec
    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-gram (better for infrequent words/products)
        epochs=epochs
    )
    
    print(f"✅ Prod2Vec trained. Vocabulary size: {len(model.wv.index_to_key)}")
    return model

def get_product_embedding(model, product_id):
    """
    Retrieves the embedding vector for a specific product.
    """
    try:
        return model.wv[str(product_id)]
    except KeyError:
        return np.zeros(model.vector_size)

# =============================================================================
# 2. MARKOV CHAINS (Transition Probabilities)
# =============================================================================

def build_markov_transition_matrices(df, state_col='FamilyLevel2'):
    """
    Builds Markov Chain transition matrices (typically per Country or Global).
    Calculates the probability of moving from Product A -> Product B.
    
    Args:
        df (pd.DataFrame): Transaction data.
        state_col (str): The level to build transitions on (e.g., 'ProductID' or 'FamilyLevel2').
        
    Returns:
        dict: A dictionary of transition matrices (one per country).
              Structure: { 'FRA': { 'Racket': {'Balls': 0.4, 'Shoes': 0.1}, ... }, ... }
    """
    print(f"🔄 Building Markov Transition Matrices on '{state_col}'...")
    
    # Ensure sorted
    df_sorted = df.sort_values(['ClientID', 'SaleTransactionDate'])
    
    # Dictionary to store transitions per country
    transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # We iterate by country to handle regional preferences
    countries = df_sorted['StoreCountry'].unique() if 'StoreCountry' in df_sorted.columns else ['Global']
    
    for country in countries:
        if country == 'Global':
            country_df = df_sorted
        else:
            country_df = df_sorted[df_sorted['StoreCountry'] == country]
            
        # Group by client and get sequence of states
        # Filter out sequences length < 2 (no transition possible)
        seqs = country_df.groupby('ClientID')[state_col].apply(list)
        seqs = seqs[seqs.apply(len) > 1]
        
        for seq in seqs:
            # Count transitions (Current -> Next)
            for i in range(len(seq) - 1):
                current_state = seq[i]
                next_state = seq[i+1]
                
                if current_state != next_state: # Optional: Ignore self-loops (re-buying same thing immediately)
                    transitions[country][current_state][next_state] += 1
                    
        # Normalize counts to probabilities
        for start_node, destinations in transitions[country].items():
            total_transitions = sum(destinations.values())
            for end_node, count in destinations.items():
                transitions[country][start_node][end_node] = count / total_transitions
                
    print(f"✅ Markov Matrices built for {len(transitions)} regions.")
    return dict(transitions)

def predict_next_state_markov(transition_matrix, current_state, top_k=3):
    """
    Predicts the top K most likely next states given a current state.
    
    Args:
        transition_matrix (dict): The dictionary of probabilities for a specific country.
        current_state (str): The product/family the user just interacted with.
        
    Returns:
        list: Top K predicted next states [(State, Probability), ...]
    """
    if current_state not in transition_matrix:
        return []
    
    # Get all possible next states and probs
    next_states = transition_matrix[current_state]
    
    # Sort by probability descending
    sorted_states = sorted(next_states.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_states[:top_k]

# =============================================================================
# 3. HYBRID RECOMMENDER UTILS
# =============================================================================

def get_similar_products_prod2vec(model, product_id, top_n=5):
    """
    Finds products semantically similar to the input product using vector space.
    """
    try:
        # returns list of (product_id, similarity_score)
        return model.wv.most_similar(str(product_id), topn=top_n)
    except KeyError:
        return []

if __name__ == "__main__":
    print("Sequence Models Library loaded.")
    print("Contains: train_prod2vec, build_markov_transition_matrices")