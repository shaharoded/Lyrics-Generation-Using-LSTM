'''
Handles the creation of dataset up to turning it into an appropriate data loader.
'''

import os
import json
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import pickle

import warnings

# Suppress the specific RuntimeWarning from pretty_midi
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*Tempo, Key or Time signature change events found on non-zero tracks.*"
)

# Local Code
from config import *
from word2vec import *
from melody2vec import *


class LyricsMelodyDataset(Dataset):
    '''
    A dataset object to process and load lyrics and melodies.
    '''
    def __init__(self, midi_dir, lyrics_file_path, failed_files_path, 
                 word2vec_path, processed_pickle_path="Data/processed_data.pkl"):
        """
        Initialize the dataset, excluding rows with failed files.
        
        Args:
            midi_dir (str): Directory containing MIDI files.
            lyrics_file_path (str): Path to the CSV file with lyrics.
            failed_files_path (list): Path to list of MIDI files that could not be processed.
            word2vec_path (str): Path to the Word2Vec bin file.
            processed_pickle_path (str): Path to cache processed data in pickle format.
        """
        self.midi_dir = midi_dir
        self.processed_pickle_path = processed_pickle_path
        self.word2vec_path = word2vec_path
        
        # Step 1: Fix MIDI files and load failed files
        fix_midi_files(ORIGINAL_MELODY_FILES, midi_dir, failed_files_path)
        self.failed_files = set()
        if os.path.exists(failed_files_path):
            with open(failed_files_path, 'r') as json_file:
                self.failed_files = set(json.load(json_file))
        
        # Step 2: Load or process dataset
        if os.path.exists(self.processed_pickle_path):
            print("[Build Status]: Loading preprocessed data...")
            with open(self.processed_pickle_path, 'rb') as f:
                saved_data = pickle.load(f)
            self.data = saved_data["data"]
            self.word2vec_path = saved_data["word2vec_path"]
            self.vocab, self.vocab_inv, self.embedding_weights = load_word2vec(word2vec_path)
            print(f"[Build Status]: Loaded processed data from {self.processed_pickle_path}.")
        else:
            print("[Build Status]: Processing lyrics and tokenizing data...")
            self.data = self.__process_data(lyrics_file_path)
            
            # Step 3: Update or load Word2Vec
            if not os.path.exists(word2vec_path.replace('.bin', '_vocab.json')):
                print("[Build Status]: Updating Word2Vec model...")
                tokenized_texts = [item["Tokenized Lyrics"] for item in self.data]
                os.makedirs(os.path.dirname(self.word2vec_path), exist_ok=True)
                update_word2vec(pretrained_path='gensim', 
                                tokenized_texts=tokenized_texts, 
                                output_path=self.word2vec_path)
            print("[Build Status]: Loading Word2Vec model...")
            self.vocab, self.vocab_inv, self.embedding_weights = load_word2vec(word2vec_path)

            # Step 5: Save data as pickle
            print("[Build Status]: Saving processed dataset...")
            self.__save_processed_data()

                
    def __process_data(self, lyrics_file_path):
        """
        Process lyrics and MIDI files, tokenize text, and filter based on Word2Vec.

        Args:
            lyrics_file_path (str): Path to the lyrics file (used to build the df)
        
        Returns:
            list: Processed dataset entries.
        """
        processed_data = []
        lyrics_df = pd.read_csv(lyrics_file_path, header=None, names=["Artist", "Song Name", "Lyrics"])
        lyrics_df['Artist'] = lyrics_df['Artist'].fillna("Unknown_Artist").str.lower()
        lyrics_df['Song Name'] = lyrics_df['Song Name'].fillna("Unknown_Song").str.lower()
        lyrics_df['Lyrics'] = lyrics_df['Lyrics'].fillna("").str.lower()

        # Get list of available MIDI files
        available_midi_files = set(
            file.lower().replace(" ", "_") for file in os.listdir(self.midi_dir) if file.endswith(".mid")
        )
        
        # Exclude rows with failed files or no matching MIDI files
        def is_valid_row(row):
            file_name = f"{row['Artist']} - {row['Song Name']}.mid".replace(" ", "_")
            if file_name not in self.failed_files and file_name not in available_midi_files:
                print(f"[Warning]: File name {file_name} was not found in folder.")
            return file_name not in self.failed_files and file_name in available_midi_files

        lyrics_df = lyrics_df[lyrics_df.apply(is_valid_row, axis=1)]
        
        # Preprocess lyrics
        lyrics_df[['Processed Lyrics', 'Tokenized Lyrics']] = lyrics_df.apply(
            lambda row: pd.Series(process_lyrics(row["Lyrics"], row["Artist"])),
            axis=1
        )

        for _, row in lyrics_df.iterrows():
            artist, song_name, original_lyrics, processed_lyrics, tokens = row["Artist"], row["Song Name"], row["Lyrics"], row["Processed Lyrics"], row["Tokenized Lyrics"]
            file_name = f"{artist} - {song_name}.mid".replace(" ", "_")
            midi_path = os.path.join(self.midi_dir, file_name)
            
            # Process MIDI
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
                melody_data = process_midi(midi, processed_lyrics)
                
            except Exception as e:
                print(f"[Warning]: Error processing MIDI file {file_name}: {e}")
                melody_data = []

            processed_data.append({
                "Artist": artist,
                "Song Name": song_name,
                "Original Lyrics": original_lyrics,
                "Processed Lyrics": processed_lyrics,
                "Tokenized Lyrics": tokens, # Embedded during model training
                "Melody Vectors": melody_data   # List of vectors per word
            })

        return processed_data

    
    def __save_processed_data(self):
            """
            Save the processed data and Word2Vec path to the pickle file.
            """
            saved_data = {
                "data": self.data,
                "word2vec_path": self.word2vec_path
            }
            with open(self.processed_pickle_path, 'wb') as f:
                pickle.dump(saved_data, f)
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]
    

class ChunkedDataset(Dataset):
    """
    Dataset that splits data from the regular dataset into fixed-size chunks.
    Needed in order to equally feed the data into a generative model.
    """
    def __init__(self, base_dataset, chunk_size, stride, indices=None):
        """
        Args:
            base_dataset (Dataset): The regular dataset containing tokenized words and melody vectors.
            chunk_size (int): Size of each chunk (number of tokens/melody vectors per chunk).
            stride (int): Step size between consecutive chunks (use chunk_size for non-overlapping chunks).
            indices (list): List of indices to select data from the base dataset. Defaults to all data.
        """
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.stride = stride
        self.indices = indices if indices else list(range(len(base_dataset)))
        self.vocab = base_dataset.vocab  # Derive vocab from base dataset
        
        # Parse tokens and melody vectors from the base dataset
        self.tokens, self.melody_vectors = self._parse_base_dataset()
        self.chunks = self.create_chunks()
    
    def _parse_base_dataset(self):
        """
        Extract tokens and melody vectors from the base dataset.

        Returns:
            tuple: (tokens, melody_vectors)
        """
        tokens = []
        melody_vectors = []
        for idx in range(len(self.base_dataset)):
            item = self.base_dataset[idx]
            token_seq, melody_seq = item["Tokenized Lyrics"], item["Melody Vectors"]
            tokens.extend(token_seq)
            melody_vectors.extend(melody_seq)
        return tokens, melody_vectors
    
    def create_chunks(self):
        chunks = []
        for i in range(0, len(self.tokens) - self.chunk_size, self.stride):
            token_chunk = self.tokens[i:i + self.chunk_size]
            melody_chunk = self.melody_vectors[i:i + self.chunk_size]
            target_chunk = self.tokens[i + 1:i + self.chunk_size + 1]
            if len(target_chunk) < self.chunk_size:
                continue  # Skip incomplete chunks at the end
            # Map tokens to indices
            token_chunk_indices = [
                self.vocab.get(token, self.vocab["<UNK>"]) for token in token_chunk
            ]
            target_chunk_indices = [
                self.vocab.get(token, self.vocab["<UNK>"]) for token in target_chunk
            ]

            chunks.append((
                torch.tensor(token_chunk_indices, dtype=torch.long),
                torch.tensor(melody_chunk, dtype=torch.float),
                torch.tensor(target_chunk_indices, dtype=torch.long),
            ))
        return chunks
    
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def save_processed_lyrics_to_csv(dataset, output_file='test_for_inspection.csv'):
    """
    Save processed lyrics and metadata to a CSV file for inspection.

    Args:
        dataset (LyricsMelodyDataset): The dataset object containing processed lyrics.
        output_file (str): Path to the output CSV file.
    """
    processed_data = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        artist = item["Artist"]
        song_name = item["Song Name"]
        original_lyrics = item["Original Lyrics"]
        processed_lyrics = item["Processed Lyrics"]
        
        processed_data.append({
            "Index": i,
            "Artist": artist,
            "Song Name": song_name,
            "Original Lyrics": original_lyrics,
            "Processed Lyrics": processed_lyrics
        })
        
    # Save to a CSV file
    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False)
    print(f"[Build Status]: Processed lyrics saved to {output_file}")
    
    
def get_data_loader(base_dataset, batch_size, val_ratio,
                    chunk_size, stride, shuffle=True):
    """
    Create DataLoaders for training and validation using the ChunkDataset.

    Args:
        base_dataset (Dataset): The base dataset containing tokenized words and melody vectors.
        chunk_size (int): Size of each chunk (number of tokens/melody vectors per chunk).
        stride (int): Step size between consecutive chunks (use chunk_size for non-overlapping chunks).
        batch_size (int): Batch size for DataLoaders.
        val_ratio (float): Proportion of dataset to use for validation. Default is 0.2 (20%).
        shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.

    Returns:
        tuple: (train_loader, val_loader) if val_ratio > 0, otherwise (train_loader,).
    """
    if val_ratio < 0.0 or val_ratio > 1.0:
        raise ValueError("val_ratio must be between 0.0 and 1.0")

    # Split the base dataset into training and validation sets
    dataset_size = len(base_dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    if shuffle:
        indices = torch.randperm(dataset_size).tolist()
    else:
        indices = list(range(dataset_size))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create Chunked Datasets
    train_chunked = ChunkedDataset(base_dataset, chunk_size=chunk_size, stride=stride, indices=train_indices)
    val_chunked = ChunkedDataset(base_dataset, chunk_size=chunk_size, stride=stride, indices=val_indices) if val_ratio > 0 else None

    # Create DataLoaders
    train_loader = DataLoader(train_chunked, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    if val_chunked:
        val_loader = DataLoader(val_chunked, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader
    else:
        return train_loader


# # Example usage - > inspect the preprocess
# if __name__ == "__main__": 
#     # Print only the first 3 samples
#     dataset = LyricsMelodyDataset(midi_dir=MELODY_FILES, 
#                                   lyrics_file_path=TRAIN_CSV,
#                                   failed_files_path=FAILED_FILES,
#                                   word2vec_path=EMBEDDING_MODEL,
#                                   embedding_dim=WORD_EMBEDDING_DIM)

           
#     # Print the total number of items in the dataset
#     print(f"[Info]: Dataset length: {len(dataset)}")
    
#     # Save dataset for inspection
#     save_processed_lyrics_to_csv(dataset)
    
#     # Ensure appropriate vectors allocation between melody and lyrics
#     for row_idx in range(len(dataset)):
#         item = dataset[row_idx]
#         song_name, melody, tokens = item['Song Name'], item['Melody Vectors'], item['Tokenized Lyrics']
#         if len(melody) != len(tokens):
#             print(f'[Warning]: n Vectors ({len(tokens)}) != n Melody ({len(melody)}) Vectors for row:', song_name)
        
#     # Print only the first 3 samples
#     for i in range(1):  # Adjust the number as needed
#         item = dataset[i]
#         melody, lyrics = item['Melody Vectors'], item['Processed Lyrics']
#         print(f"Sample {i + 1}:")
#         words = lyrics.split()[:20]
#         for idx, word in enumerate(words): 
#             # Display the first 10 melody notes and lyrics tokens
#             print("Word: ", words[idx])
#             print("Melody: ", melody[idx])
#             print("-" * 50)  # Separator for readability