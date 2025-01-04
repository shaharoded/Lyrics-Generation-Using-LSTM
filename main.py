import os
import torch
from torch.utils.data import DataLoader

# Local Code
from dataset import LyricsMelodyDataset, ChunkedDataset
from lstm import LyricsGenerator
from config import *

def build_dataset():
    """
    Build the dataset for lyrics and melody.
    """
    print("[Option 1]: Building dataset...")
    dataset = LyricsMelodyDataset(midi_dir=MELODY_FILES, 
                                lyrics_file_path=TRAIN_CSV,
                                failed_files_path=FAILED_FILES,
                                word2vec_path=EMBEDDING_MODEL)
    print(f"[Build Status]: Dataset built with {len(dataset)} samples.")
    return dataset


def train_model(dataset):
    """
    Train the model using dataset and config.py parameters.
    """
    print("[Option 2]: Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LyricsGenerator(dataset=dataset,
                            word_embedding_dim=300, 
                            melody_dim=11, 
                            hidden_dim=128,
                            num_layers=1,
                            bidirectional_melody=BIDIRECTIONAL,
                            dropout=DROPOUT,
                            use_attention=USE_ATTENTION)
    name = f"{'attention' if USE_ATTENTION else 'regular'}_{'bidirectional' if BIDIRECTIONAL else 'unidirectional'}"
    model.train_model(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        stride=STRIDE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        teacher_forcing_ratio=TEACHERS_FORCING,
        epochs=NUM_EPOCHS,
        device=device,
        val_ratio=VAL_RATIO,
        checkpoint_path=BEST_MODEL.format(name)
    )


def generate_text(train_dataset, song_name, start_token=BOT_TOKEN):
    """
    Generate text for a given song using trained model and config.py parameters.
    
    Args:
        train_dataset (Dataset): The train dataset. Used to access the Word2Vec Vocab.
        song_name (str): A name of a song from the test set.
        start_token (str): The starting token for generation. Defaults to <BeginningOfText>.
    """
    print("[Option 3]: Generating text...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LyricsGenerator(dataset=train_dataset, 
                            word_embedding_dim=300, 
                            melody_dim=11, 
                            hidden_dim=128,
                            num_layers=1,
                            bidirectional_melody=BIDIRECTIONAL,
                            dropout=DROPOUT,
                            use_attention=USE_ATTENTION)
    
    name = f"{'attention' if USE_ATTENTION else 'regular'}_{'bidirectional' if BIDIRECTIONAL else 'unidirectional'}"
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.load_state_dict(torch.load(
        BEST_MODEL.format(name),
        weights_only=True,
        map_location=map_location
    ))
    model.eval()

    generated_text = model.generate_text(
        song_name=song_name,
        start_token=start_token,
        max_length=MAX_OUTPUT_LENGTH,
        device=device,
        temperature=TEMEPRATURE,
        test_csv=TEST_CSV
    )
    print("Generated Text:")
    print(generated_text)


def main():
    dataset = None
    while True:
        print("\nMain Menu")
        print("1 - Build Train Dataset Object")
        print("2 - Train a Model")
        print("3 - Generate Text")
        print("4 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            dataset = build_dataset()
        elif choice == '2':
            if dataset is None:
                print("Please build a dataset first (Option 1).")
            else:
                train_model(dataset)
        elif choice == '3':
            if dataset is None:
               print("Please build a dataset first (Option 1).")
            else: 
                song_name = input("Enter the song name to generate text for: ")
                start_token = input("Enter the starting token for text generation or None: ")
                start_token = BOT_TOKEN if str.lower(start_token) == 'none' else start_token
                generate_text(train_dataset=dataset,
                              song_name=song_name,
                              start_token=start_token)
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()