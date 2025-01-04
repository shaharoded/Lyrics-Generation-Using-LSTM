'''
Functions for vectorization of the melody
'''
import pretty_midi
from pretty_midi import program_to_instrument_name
import numpy as np
import re
import os
import json

# Local Code
from config import (SPECIAL_TOKENS, REPLACEMENTS, EOT_TOKEN, BOT_TOKEN, 
                    CHORUS_TOKEN, NEWLINE_TOKEN)

# Define a list of instrument categories
INSTRUMENT_CATEGORIES = ["Keyboard", "Strings", "Percussion", "Brass", "Woodwind", "Other"]
CATEGORY_TO_INDEX = {category: idx for idx, category in enumerate(INSTRUMENT_CATEGORIES)}


def map_instrument(program):
    instrument_name = program_to_instrument_name(program)
    if "Piano" in instrument_name or "Keyboard" in instrument_name:
        category = "Keyboard"
    elif "String" in instrument_name or "Guitar" in instrument_name:
        category = "Strings"
    elif "Drum" in instrument_name or "Percussion" in instrument_name:
        category = "Percussion"
    elif "Brass" in instrument_name:
        category = "Brass"
    elif "Reed" in instrument_name or "Flute" in instrument_name:
        category = "Woodwind"
    else:
        category = "Other"

    # Create one-hot encoding
    one_hot = [0] * len(INSTRUMENT_CATEGORIES)
    one_hot[CATEGORY_TO_INDEX[category]] = 1
    return one_hot


def fix_midi_files(input_dir, output_dir, failed_files_path):
    """
    Fix MIDI files by handling meta-events and excluding invalid files.

    Args:
        input_dir (str): Directory containing original MIDI files.
        output_dir (str): Directory to save fixed MIDI files.
        failed_files_path (str): File path to save failed files for dataset initialization.

    Returns:
        None. Processes MIDI files and logs invalid ones.
    """
    if os.path.exists(output_dir) and os.path.exists(failed_files_path):
        print('[Build Status]: MIDI files repository is up-to-date. Moving on...')
        return

    os.makedirs(output_dir, exist_ok=True)
    failed_files = []
    
    # Create a dictionary for case-insensitive filename matching
    lower_upper_files = get_lower_upper_dict(input_dir)

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".mid"):
            continue

        # Standardize the filename and validate existence
        standardized_name = file_name.lower().replace(' ', '_')
        if standardized_name not in lower_upper_files:
            print(f"[Warning]: File {standardized_name} does not exist in the directory.")
            failed_files.append(file_name)
            continue

        original_name = lower_upper_files[standardized_name]
        input_path = os.path.join(input_dir, original_name)
        output_path = os.path.join(output_dir, original_name)

        try:
            # Load the MIDI file using PrettyMIDI
            midi = pretty_midi.PrettyMIDI(input_path)

            # Remove invalid or empty melody files
            if midi.get_end_time() == 0 or not any(len(inst.notes) > 0 for inst in midi.instruments):
                print(f"[Warning]: Skipping {file_name} due to empty melody or zero duration.")
                failed_files.append(file_name)
                continue

            # Add meta-events to the first instrument (if needed)
            first_instrument = pretty_midi.Instrument(program=0, is_drum=False)
            for time_sig in midi.time_signature_changes:
                midi_time = time_sig.time
                first_instrument.notes.append(pretty_midi.Note(
                    velocity=0, pitch=0, start=midi_time, end=midi_time + 0.1
                ))
            for tempo in midi.get_tempo_changes()[1]:
                midi_time = midi.get_tempo_changes()[0][0]
                first_instrument.notes.append(pretty_midi.Note(
                    velocity=0, pitch=0, start=midi_time, end=midi_time + 0.1
                ))

            midi.instruments.insert(0, first_instrument)

            # Save the fixed MIDI file
            midi.write(output_path)

        except Exception as e:
            print(f"[Warning]: Error processing {file_name}: {e}")
            failed_files.append(file_name)

    # Save the failed files to a JSON file
    with open(failed_files_path, 'w') as json_file:
        json.dump(failed_files, json_file, indent=4)
    print(f"[Build Status]: Saved failed files list to {failed_files_path}.")


def get_lower_upper_dict(midi_folder):
    """
    Create a mapping between lowercase filenames and their original case-sensitive counterparts.

    Args:
        midi_folder (str): Path to the folder containing MIDI files.

    Returns:
        dict: Mapping from lowercase filenames to original filenames.
    """
    lower_upper_files = {}
    for file_name in os.listdir(midi_folder):
        if file_name.endswith(".mid"):
            lower_upper_files[file_name.lower()] = file_name
    return lower_upper_files


def get_word_durations(processed_lyrics, total_duration):
    """
    Calculate the start and end times for each word in the lyrics, ignoring special tokens for duration calculations.

    Args:
        processed_lyrics (str): The lyrics of the song, split into rows.
        total_duration (float): Total duration of the MIDI file.

    Returns:
        list of tuple: List of (start_time, end_time, word) for each word in the lyrics.
    """
    word_durations = []
    processed_lyrics = processed_lyrics.strip()
    rows = [segment for segment in processed_lyrics.split(NEWLINE_TOKEN)] # Split based on NEWLINE, keeping the special token.
    rows = [row + f" {NEWLINE_TOKEN}" if not row.endswith(EOT_TOKEN) else row for row in rows]  # Ensure <BR> at end
    non_empty_rows = [row for row in rows if row.strip() and row.strip() not in SPECIAL_TOKENS]
    row_duration = total_duration / len(non_empty_rows) if non_empty_rows else total_duration  # Time per valid row

    current_time = 0
    for row in rows:
        words = row.split() # Get all row tokens
        
        # Exclude special tokens for duration calculation
        regular_words = [word for word in words if word not in SPECIAL_TOKENS]
        word_duration = row_duration / len(regular_words) if regular_words else 0

        for word in words:
            start_time = current_time
            if word in SPECIAL_TOKENS:
                # Assign zero-duration for special tokens
                end_time = start_time
            else:
                end_time = start_time + word_duration

            word_durations.append((start_time, end_time, word))
            current_time = end_time
    return word_durations


def process_midi(midi, processed_lyrics, aggregation_method='max'):
    """
    Extract word-level melody features from the MIDI file, aggregating by word time span.

    Args:
        midi (PrettyMIDI): The MIDI file.
        processed_lyrics (str): Processed Lyrics to calculate time spans per word.
        aggregation_method (str): Aggregation method ('avg' or 'max').

    Returns:
        list: Aggregated melody feature vectors size 11 per word in the lyrics.
    """
    word_durations = get_word_durations(processed_lyrics, midi.get_end_time())
    features_per_word = []

    # Extract tempo and key
    _, tempi = midi.get_tempo_changes()
    tempo = tempi[0] if len(tempi) > 0 else 120  # Default to 120 BPM
    tempo_normalized = tempo / 300  # Normalize tempo

    key_changes = midi.key_signature_changes
    key = key_changes[0].key_number if key_changes else 0  # Default to C major
    
    # Define neutral vector
    neutral_vector_template = [1, 0, 0, 0, 0, key, tempo_normalized, 0, 0, 0, 0]

    for start_time, end_time, word in word_durations:
        if start_time == end_time:
            # Specifically mark which special token this is
            neutral_vector = neutral_vector_template.copy()
            neutral_vector[0] = SPECIAL_TOKENS.index(word) + 1 / len(SPECIAL_TOKENS)
            # Assign neutral vector for special tokens
            features_per_word.append(neutral_vector)
            continue
        # Initialize features for the time span
        pitches, velocities, durations, programs = [], [], [], []
        num_instruments, num_notes = 0, 0
        has_drums, has_piano = 0, 0

        # Extract features for notes within the time span
        for instrument in midi.instruments:
            instrument_category = map_instrument(instrument.program)
            if instrument.is_drum:
                has_drums = 1
            if instrument_category[0] == 1:  # Assume piano maps to [1, 0, 0, ...]
                has_piano = 1

            in_range = False
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    in_range = True
                    num_notes += 1
                    pitches.append(note.pitch)
                    velocities.append(note.velocity)
                    durations.append(note.end - note.start)
                    programs.append(instrument.program)

            if in_range:
                num_instruments += 1

        # Aggregate features
        if num_notes > 0:
            pitch_agg = np.mean(pitches) if aggregation_method == 'avg' else np.max(pitches)
            velocity_agg = np.mean(velocities) if aggregation_method == 'avg' else np.max(velocities)
            duration_agg = np.mean(durations) if aggregation_method == 'avg' else np.max(durations)
            program_agg = np.mean(programs) / 127 if programs else 0  # Normalize program
        else:
            pitch_agg = velocity_agg = duration_agg = program_agg = 0

        density = num_notes / (end_time - start_time) if end_time > start_time else 0

        # Combine features
        features = [
            0,  # Symbol for is_neutral_vector
            pitch_agg / 127,  # Normalize pitch
            velocity_agg / 127,  # Normalize velocity
            duration_agg / midi.get_end_time(),  # Normalize duration
            program_agg,  # Normalized program
            key,
            tempo_normalized,
            density / 179,  # Max Density based on analysis
            num_instruments / 16,   # Midi file allows for 16 tracks max == 16 instruments.
            has_drums,
            has_piano,
        ]
        features_per_word.append(features)

    return features_per_word
    