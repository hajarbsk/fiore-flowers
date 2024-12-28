import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculer la similarité cosinus entre deux vecteurs"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
