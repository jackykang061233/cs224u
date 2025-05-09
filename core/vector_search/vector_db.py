from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.vector_search.vector_embedding import vector_embedding
from sentence_transformers import SentenceTransformer
import uuid
import torch

class Vector_DB:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        
    def create_collection(self, collection_name, size=384):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.DOT),
        )
        
    def add_vectors(self, collection_name, sentences):
        sentences_to_embedd = sentences['sentences']
        payloads = sentences['payloads']

        embeddings, embeddings_shape = vector_embedding(sentences_to_embedd)

        # Ensure collection exists
        if not self.client.collection_exists(collection_name=collection_name):
            self.create_collection(collection_name, size=embeddings_shape)

        # Convert embeddings to lists if they're tensors
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().tolist()

        # Prepare points
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload=payloads[i]
            )
            for i in range(len(embeddings))
        ]

        # Upload points
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    
    def search(self, collection_name, query):
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query,
            query_filter=Filter(
                must=[FieldCondition(key="city", match=MatchValue(value="London"))]
            ),
            with_payload=True,
            limit=5,
        ).points
        return search_result

