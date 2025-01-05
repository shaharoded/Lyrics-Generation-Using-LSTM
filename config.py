# Data Paths:
ORIGINAL_MELODY_FILES = 'Data/midi_files/'
MELODY_FILES = 'Data/fixed_midi_files/'
TRAIN_CSV = 'Data/lyrics_train_set.csv'
TEST_CSV = 'Data/lyrics_test_set.csv'
FAILED_FILES = 'Data/failed_files.json'
EMBEDDING_MODEL = "Embedding Model/updated_word2vec.bin"
BEST_MODEL = 'Trained Models/best_{}_model.pth'

# Special Tokens:
EOT_TOKEN = '<EndOfText>'
BOT_TOKEN = '<BeginningOfText>'
CHORUS_TOKEN = '<Chorus>'
NEWLINE_TOKEN = '<BR>'
UNKNOWN_TOKEN = '<UNK>'
REPLACEMENTS = {    # char_to_replace: replacement, is_replacement_a_token
    "{": ("(", True),
    "}": (")", True),
    "[": ("(", True),
    "]": (")", True),
    "-": (" ", False),
    ";": (",", True),
    "-": (" ", False),  # Replace dashes with spaces for better separation
    "in'": ("ing", False),  # Standardize common informal contractions
    "&": (NEWLINE_TOKEN, True),  # Replace '&' with newline for structure
    "‘": ("'", True),  # Standardize quotes
    "’": ("'", True),  # Replace curly quotes with straight quotes
    "“": ('"', True),  # Standardize double quotes
    "”": ('"', True),  # Replace curly double quotes
    "—": (" ", False),  # Replace em-dash with space
    "…": ("...", True),  # Standardize ellipses
    "\u200b": ("", False),  # Remove zero-width space characters
    "\xa0": (" ", False),  # Replace non-breaking spaces
    "\t": (" ", False),  # Replace tabs with a single space
}
SPECIAL_TOKENS = sorted(list({BOT_TOKEN, NEWLINE_TOKEN, CHORUS_TOKEN, EOT_TOKEN, UNKNOWN_TOKEN} | {value[0] for _, value in REPLACEMENTS.items() if value[1]}))

# Training Parameters
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
TEACHERS_FORCING = 0.3
VAL_RATIO = 0.2
CHUNK_SIZE = 50     # 50 processed tokens in each window
STRIDE = 25     # 25 tokens overlap between one window and the next
BATCH_SIZE = 16
EARLY_STOP = 5
DROPOUT = 0.5
NUM_EPOCHS = 40
BIDIRECTIONAL = True    # Choose between True / False to modify architecture
USE_ATTENTION = False   # Choose between True / False to modify architecture

# Text Generation
TEMEPRATURE = 1.5   # Values above 1.0 create a more 'creative' model.
MAX_OUTPUT_LENGTH = 250     # Limit the output length if hasn't stop on it's own
