from core.vector_search.vector_embedding import vector_embedding
import torch
import torch
from sentence_transformers import util

def rank_places(data, keywords=None, weights=None):
    """
    Rank places based on review similarity (if keywords provided), rating, and total travel time.
    
    Args:
        data (list): List of dictionaries with 'place' (str), 'reviews' (list of dicts with 'text', etc.),
                     'rating' (float/int, 1-5), and 'total_travel_time' (float/int, minutes).
        keywords (str or list, optional): Keywords for similarity search (e.g., "cozy friendly comfortable").
                                         If None, empty string, or empty list, similarity is ignored.
        weights (dict, optional): Weights for scoring components. Defaults depend on keywords:
                                 With keywords: {'similarity': 0.5, 'rating': 0.3, 'travel_time': 0.2}.
                                 Without keywords: {'rating': 0.6, 'travel_time': 0.4}.
    
    Returns:
        list: List of dictionaries with original place data, plus 'combined_score' and
              'average_similarity' (if keywords provided), sorted by combined_score.
    """
    # Determine if keywords are provided
    use_keywords = keywords and (isinstance(keywords, str) and keywords.strip()) or (isinstance(keywords, list) and keywords)
    
    # Set default weights based on whether keywords are used
    if weights is None:
        if use_keywords:
            weights = {'similarity': 0.5, 'rating': 0.3, 'travel_time': 0.2}
        else:
            weights = {'rating': 0.6, 'travel_time': 0.4}
    
    # Validate weights
    total_weight = sum(weights.values())
    if not (0.99 <= total_weight <= 1.01):  # Allow small float errors
        raise ValueError("Weights must sum to 1")
    
    # Prepare keywords if provided
    if use_keywords:
        if isinstance(keywords, str):
            keywords = [keywords]
        # Compute embeddings for keywords
        keyword_embeddings, _ = vector_embedding(keywords)
        if torch.cuda.is_available():
            keyword_embeddings = keyword_embeddings.to('cuda')
    
    # Process reviews and calculate scores
    place_scores = []
    
    for place_data in data:
        place_name = place_data.get("name")
        reviews = place_data.get("reviews", [])
        rating = place_data.get("rating", 0)
        travel_time = place_data.get("travel_time", float('inf'))
        
        if not place_name or rating <= 0 or travel_time <= 0:
            continue  # Skip invalid or incomplete data
        
        # Initialize scores
        avg_similarity = 0
        if use_keywords:
            # Extract review texts
            review_texts = [review.get("text", "") for review in reviews if review.get("text")]
            if not review_texts:
                continue  # Skip if no valid review texts
            
            # Compute similarity score
            review_embeddings, _ = vector_embedding(review_texts)
            similarities = []
            for review_emb in review_embeddings:
                if torch.cuda.is_available():
                    review_emb = review_emb.to('cuda')
                # Average similarity across keywords
                review_similarities = [util.cos_sim(review_emb, kw_emb).item() for kw_emb in keyword_embeddings]
                similarities.append(sum(review_similarities) / len(review_similarities))
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Normalize rating (1-5 to 0-1)
        rating_score = rating / 5.0
        
        # Normalize travel time (shorter is better, 0-1)
        travel_time_score = 1 / (1 + travel_time)
        
        # Compute combined score
        if use_keywords:
            combined_score = (
                weights['similarity'] * avg_similarity +
                weights['rating'] * rating_score +
                weights['travel_time'] * travel_time_score
            )
        else:
            combined_score = (
                weights['rating'] * rating_score +
                weights['travel_time'] * travel_time_score
            )
        
        # Copy place_data and add scores
        place_dict = place_data.copy()
        place_dict["combined_score"] = combined_score
        if use_keywords:
            place_dict["average_similarity"] = avg_similarity
        
        place_scores.append(place_dict)
    
    # Sort by combined_score in descending order
    place_scores.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return place_scores
    
