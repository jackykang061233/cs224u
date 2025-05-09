from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
# from qdrant_client.models import PointStruct
# from qdrant_client.models import Filter, FieldCondition, MatchValue
# from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range

from core.vector_search.vector_embedding import vector_embedding
from sentence_transformers import SentenceTransformer
import uuid
import torch
import math
import numpy as np

class Vector_DB:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        
    def create_collection(self, collection_name, size=384):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.DOT),
        )
        
    def add_vectors(self, collection_name, place_data):
        # Extract review texts for embedding
        review_texts = [review['text'] for review in place_data['reviews']]
        
        # If no reviews, skip or handle accordingly
        if not review_texts:
            print(f"No reviews for {place_data.get('name', 'unknown place')}, skipping.")
            return

        # Prepare metadata (exclude reviews since they are embedded)
        metadata = {key: value for key, value in place_data.items() if key != 'reviews'}
        # Include review-specific metadata
        for i, review in enumerate(place_data['reviews']):
            metadata[f'review_{i}_text'] = review['text']
            metadata[f'review_{i}_author'] = review['author']
            metadata[f'review_{i}_rating'] = review['rating']
            metadata[f'review_{i}_time'] = review['time']
            metadata[f'review_{i}_timestamp'] = review['timestamp']

        # Generate embeddings for reviews
        embeddings, embeddings_shape = vector_embedding(review_texts)

        # Convert embeddings to numpy array if they're tensors
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()

        # Aggregate embeddings by taking the mean
        aggregated_embedding = np.mean(embeddings, axis=0)

        # Ensure collection exists
        if not self.client.collection_exists(collection_name=collection_name):
            self.create_collection(collection_name, size=embeddings_shape)

        # Prepare single point for the place
        point = PointStruct(
            id=str(uuid.uuid4()),  # Unique ID for the place
            vector=aggregated_embedding.tolist(),  # Convert to list
            payload=metadata  # All metadata, including review details
        )

        # Upload point
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        print(f"Successfully added place '{place_data.get('name', 'unknown')}' with aggregated embedding to Qdrant collection '{collection_name}'")
        
    def find_nearby_points(self, collection_name, latitude, longitude, radius_km=1.0, query_text=None, limit=10):
        """
        Find points within a given radius (in km) of the specified coordinates.
        Optionally, combine with vector search if query_text is provided.
        
        Args:
            collection_name (str): Name of the Qdrant collection.
            latitude (float): Latitude of the query point.
            longitude (float): Longitude of the query point.
            radius_km (float): Radius in kilometers to search within.
            query_text (str, optional): Text to perform vector search on reviews.
            limit (int): Maximum number of results to return.
        
        Returns:
            List of tuples (payload, score) for matching points.
        """
        # Calculate bounding box for the radius
        # Approximate 1 degree of latitude/longitude to kilometers
        lat_degree = 111.0  # 1 degree of latitude ~ 111 km
        lon_degree = 111.0 * math.cos(math.radians(latitude))  # Adjust longitude based on latitude

        lat_delta = radius_km / lat_degree
        lon_delta = radius_km / lon_degree

        lat_min = latitude - lat_delta
        lat_max = latitude + lat_delta
        lon_min = longitude - lon_delta
        lon_max = longitude + lon_delta

        # Create payload filter for location
        location_filter = Filter(
            must=[
                FieldCondition(
                    key="location[0]",  # Latitude
                    range=Range(gte=lat_min, lte=lat_max)
                ),
                FieldCondition(
                    key="location[1]",  # Longitude
                    range=Range(gte=lon_min, lte=lon_max)
                )
            ]
        )

        if query_text:
            # Perform vector search with payload filter
            embeddings, _ = vector_embedding([query_text])
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().tolist()
            query_vector = embeddings[0]
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=location_filter,
                limit=limit,
                with_payload=True
            )
        else:
            # Perform scroll query with payload filter (no vector search)
            search_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=location_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]  # Extract points from (points, next_page_offset)

        return [(result.payload, result.score if query_text else None) for result in search_result]

