from sentence_transformers import SentenceTransformer
import torch

def vector_embedding(sentences):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        if torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
            embeddings = model.encode(sentences, convert_to_tensor=True, device='cuda')
        else:
            embeddings = model.encode(sentences)
        embeddings_shape = embeddings.shape[1]
        return embeddings, embeddings_shape