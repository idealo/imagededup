from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn


def cosine_similarity(x, y=None):
    return cosine_similarity_sklearn(X=x, Y=y)
