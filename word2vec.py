'''
Functions for creating a custom Word2Vec model, text preprocess and vectorization of words.
'''

from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load
import torch
import numpy as np
import re
import json
import contractions
from tqdm import tqdm
import warnings

# Suppress specific warnings from gensim
warnings.filterwarnings("ignore", category=UserWarning, module="gensim")

# Local Code
from config import (SPECIAL_TOKENS, REPLACEMENTS, EOT_TOKEN, BOT_TOKEN, 
                    CHORUS_TOKEN, NEWLINE_TOKEN, UNKNOWN_TOKEN)



def update_word2vec(pretrained_path, tokenized_texts, output_path, vector_size=300, min_count=10):
    """
    Fine-tune a pre-trained Word2Vec model by creating a reduced embedding matrix 
    that includes only tokens present in the dataset and a special <UNK> token.

    Args:
        pretrained_path (str): Path to the pre-trained Word2Vec model or 'gensim' to download one.
        tokenized_texts (list of list of str): Tokenized dataset (list of tokenized sentences).
        output_path (str): Path to save the updated Word2Vec model.
        vector_size (int): Size of the embeddings, set to 300 by default.
        min_count (int): Minimum occurrences of a token to be added. Use to filter typos and over specific tokens.

    Returns:
        None. Updated vectors were saved in output_path and will be accessed using the dataset object.
    """
    # Load pre-trained Word2Vec
    print("[Build Status]: Loading pre-trained off-the-shelf Word2Vec model...")
    if pretrained_path == 'gensim':
        word2vec = gensim_load('word2vec-google-news-300')
    else:
        word2vec = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    print(f"[Build Status]: Loaded pre-trained model with {len(word2vec.index_to_key)} tokens.")

    # Flatten tokenized_texts and find unique tokens
    all_tokens = [token for text in tokenized_texts for token in text]
    token_counts = {token: all_tokens.count(token) for token in set(all_tokens)}

    # Identify tokens to keep
    tokens_to_keep = [
        token for token, count in token_counts.items()
        if count >= min_count or token in word2vec
    ]
    tokens_to_keep = list(set(tokens_to_keep))  # Ensure uniqueness
    print(f"[Build Status]: Reducing to {len(tokens_to_keep)} tokens (min count: {min_count}).")

    # Create a new Word2Vec model with only the required tokens
    print("[Build Status]: Creating reduced embedding matrix...")
    reduced_vectors = np.zeros((len(tokens_to_keep) + 1, vector_size))  # +1 for <UNK> token
    reduced_vocab = {UNKNOWN_TOKEN: 0}  # Map <UNK> token to index 0

    for idx, token in enumerate(tokens_to_keep, start=1):  # Start indexing at 1
        if token in word2vec:
            reduced_vectors[idx] = word2vec[token]
        else:
            reduced_vectors[idx] = np.random.uniform(-0.1, 0.1, vector_size)
        reduced_vocab[token] = idx

    # Add random embedding for <UNK>
    reduced_vectors[0] = np.random.uniform(-0.1, 0.1, vector_size)
    
    # Convert to PyTorch tensor
    embedding_weights = torch.tensor(reduced_vectors, dtype=torch.float32)

    # Save the reduced vocab and embeddings
    print(f"[Build Status]: Saving reduced embedding model to {output_path}...")
    with open(output_path.replace('.bin', '_vocab.json'), 'w') as vocab_file:
        json.dump(reduced_vocab, vocab_file)
    torch.save(embedding_weights, output_path.replace('.bin', '_embeddings.pt'))
    print("[Build Status]: Reduced Word2Vec model saved successfully.")


def load_word2vec(word2vec_path):
    """
    Load Word2Vec embeddings and vocabulary from paths derived from the original model path.

    Args:
        word2vec_path (str): Path to the original Word2Vec binary file.

    Returns:
        tuple: (vocab, embedding_weights)
            - vocab (dict): Mapping of word to index.
            - vocab_inv (dict): Mapping of index to word.
            - embedding_weights (torch.Tensor): Pre-trained embedding weights.
    """
    # Infer paths
    vocab_path = word2vec_path.replace('.bin', '_vocab.json')
    embeddings_path = word2vec_path.replace('.bin', '_embeddings.pt')

    # Load the vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_inv = {idx: word for word, idx in vocab.items()}
    # Load the embedding weights
    embedding_weights = torch.load(embeddings_path, weights_only=True)
    
    # Verify weights only (optional)
    if not isinstance(embedding_weights, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor for embeddings, got {type(embedding_weights)}")

    print(f"[Build Status]: Loaded Word2Vec with {len(vocab)} tokens from {word2vec_path}.")
    return vocab, vocab_inv, embedding_weights
    
  
def process_lyrics(lyrics, artist):
    '''
    Return normalized lyrics, number of rows, and number of words.
    Works on a row level.
    '''        
    # Replace characters based on the mapping
    for old_char, (new_char, _) in REPLACEMENTS.items():
        lyrics = lyrics.replace(old_char, new_char) 
    
    # Expand constractions in lyrics to unify tokens
    lyrics = contractions.fix(lyrics)
            
    # Replace any variations of (chorus+) with <Chorus>
    lyrics = re.sub(r"\(?\bchorus\b.*?\)?", CHORUS_TOKEN, lyrics, flags=re.IGNORECASE)
    # Emove repeats from text (x2)
    lyrics = re.sub(r"\(\s*[xX]\d+\s*\)", "", lyrics)
    # Attend edge cases
    lyrics = lyrics.replace('aingt', "ain't").replace("'s", " s")
    
    # Remove artist name from the end of lyrics
    if lyrics.strip().endswith(artist.strip()):
        lyrics = lyrics[: -len(artist)].strip()
    
    # Clean misplaced neutral tokens
    if lyrics.strip().endswith(NEWLINE_TOKEN):  # Check if it ends with <BR>
        lyrics = lyrics[:-len(NEWLINE_TOKEN)].strip()  # Remove <BR> and trailing spaces
    
    # Add BOT and EOT tokens
    processed_lyrics = f"{BOT_TOKEN} {lyrics} {EOT_TOKEN}"
    tokens = tokenize_lyrics(processed_lyrics)
    processed_lyrics = ' '.join(tokens)
    
    return processed_lyrics, tokens


def tokenize_lyrics(lyrics, word2vec=None):
    """
    Tokenize lyrics while respecting REPLACEMENTS.
    Keeps specified replacement tokens as independent tokens and ignores spaces or empty strings.
    Will filter tokens not in word2vec if was supplied with a reference model.

    Args:
        lyrics (str): Semi-Preprocessed lyrics string.
        word2vec (KeyedVectors): Used to verify the existance of the token in the embedding method.
                                Since Word2Vec is calculated using the Processed Lyrics, no need to 
                                pass this during the training process, only predictions.

    Returns:
        list: List of tokens.
    """
    tokens = []

    # Apply REPLACEMENTS to normalize the lyrics
    for old_char, (replacement, is_token) in REPLACEMENTS.items():
        if is_token:
            # Add spaces around replacements to ensure token separation
            lyrics = lyrics.replace(old_char, f" {replacement} ")
        else:
            lyrics = lyrics.replace(old_char, replacement)

    # Split lyrics by whitespace and punctuation, retaining meaningful tokens
    split_pattern = r'(\s+|[?.,:!()"])'
    raw_tokens = re.split(split_pattern, lyrics)
    raw_tokens = [token for token in raw_tokens if not token.isspace() and token]

    # Process tokens: keep meaningful replacements and remove empty strings
    for token in raw_tokens:
        token = token.strip()  # Remove extra spaces
        if not word2vec or token in word2vec:
            tokens.append(token)

    return tokens