import pickle
import numpy as np
import torch

def get_book_authors(author):
    """
    Gets author names by removing extra spaces and handling multiple authors.
    """
    author = str(author)

    # Remove extra spaces between words
    author = ' '.join(author.split())
    
    # Split by '|' if there are multiple authors and return as a list
    return author.split('|')

def normalize_author_name(author):
    """
    Normalizes author names by removing extra spaces and lowercasing.
    """
    author = str(author)

    # Remove extra spaces between words
    author = ' '.join(author.split())

    # Lowercase
    author = author.lower()

    return author

def get_author_embedding(author2embedding, author):
    """
    Fetches embedding for an author and returns it.
    If the author is missing from the embeddings, it will be ignored.
    """
    normalized_author2embedding = {normalize_author_name(author): embedding for author, embedding in author2embedding.items()}
    normalized_author = normalize_author_name(author)
    if normalized_author in normalized_author2embedding:
        return normalized_author2embedding[normalized_author]
    else:
        return None

def get_authors_embedding(author2embedding, book_authors):
    """
    Fetches embeddings for a list of authors and returns the average embedding.
    If any author is missing from the embeddings, it will be ignored.
    """
    authors = get_book_authors(book_authors)

    # Get embedding for each author
    for author in authors:
        embedding = get_author_embedding(author2embedding, author)
        if embedding is not None:
            return torch.tensor(embedding.astype(np.float32))
    
    return torch.tensor(np.zeros(200).astype(np.float32))

    
    


