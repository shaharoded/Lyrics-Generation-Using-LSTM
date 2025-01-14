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
LEARNING_RATE = 1e-3    # Initial learning rate - decreases using scheduler
WEIGHT_DECAY = 1e-5
TEACHERS_FORCING = 0.5  # Initial ratio - decreases across epochs
VAL_RATIO = 0.2
CHUNK_SIZE = 50     # 50 processed tokens in each window
STRIDE = 25     # 25 tokens overlap between one window and the next
BATCH_SIZE = 16
EARLY_STOP = 5  # Relatively high due to some instability using penalties
DROPOUT = 0.5   # Only for linear layers
GENERATION_PENALTY_WEIGHT = 0.3 # Penalty added to the loss function under predefined conditions in the train function.
NUM_EPOCHS = 40
HIDDEN_SIZE = 256
BIDIRECTIONAL = False    # Choose between True / False to modify architecture
USE_ATTENTION = False   # Choose between True / False to modify architecture

# Text Generation
TEMEPRATURE = 0.8   # Values above 1.0 create a more 'creative' model.
MAX_OUTPUT_LENGTH = 250     # Limit the output length if hasn't stop on it's own
