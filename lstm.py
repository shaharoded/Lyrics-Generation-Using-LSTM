'''
An appropriate LSTM model for the task, allows for both bidirectional feed of the 
melody information or unidirectional usage.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from gensim.models import KeyedVectors
import time
import random
import re

# Local Code
from config import (EOT_TOKEN, BOT_TOKEN, CHORUS_TOKEN, NEWLINE_TOKEN, UNKNOWN_TOKEN)
from dataset import *
from word2vec import *
from melody2vec import *


class Attention(nn.Module):
    '''
    Optional addition to the LSTM architecture.
    '''
    def __init__(self, query_dim, key_dim, value_dim, output_dim):
        """
        Implements a basic attention mechanism.

        Args:
            query_dim (int): Dimension of the query vector.
            key_dim (int): Dimension of the key vector.
            value_dim (int): Dimension of the value vector.
            output_dim (int): Dimension of the output after applying attention.
        """
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(query_dim, output_dim)
        self.key_projection = nn.Linear(key_dim, output_dim)
        self.value_projection = nn.Linear(value_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values):
        """
        Forward pass for the attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_query, query_dim).
            keys (torch.Tensor): Key tensor of shape (batch_size, seq_len_keys, key_dim).
            values (torch.Tensor): Value tensor of shape (batch_size, seq_len_keys, value_dim).

        Returns:
            torch.Tensor: Context vector of shape (batch_size, seq_len_query, output_dim).
        """            
        # Project inputs
        query_proj = self.query_projection(query)  # (batch_size, seq_len_query, output_dim)
        keys_proj = self.key_projection(keys)      # (batch_size, seq_len_keys, output_dim)
        values_proj = self.value_projection(values)  # (batch_size, seq_len_keys, output_dim)
        # Compute attention scores
        scores = torch.matmul(query_proj, keys_proj.transpose(-2, -1))  # (batch_size, seq_len_query, seq_len_keys)
        attention_weights = self.softmax(scores)  # (batch_size, seq_len_query, seq_len_keys)
        
        # Compute context vector
        context = torch.matmul(attention_weights, values_proj)  # (batch_size, seq_len_query, output_dim)
        return context  # (batch_size, seq_len_query, output_dim)
    

class LyricsGenerator(nn.Module):
    def __init__(self, dataset, word_embedding_dim, 
                 melody_dim, hidden_dim, 
                 bidirectional_melody, dropout, use_attention):
        """
        Initialize the LyricsGenerator, designated to generate song lyrics based on melody.
        The model is a 2 layer LSTM model, one for processing the melody and the other to process text.

        Args:
            dataset (LyricsMelodyDataset): The dataset object containing the Word2Vec model and vocabulary.
            melody_dim (int): Dimensionality of melody vectors.
            hidden_dim (int): Hidden state size for RNNs.
            bidirectional_melody (bool): Whether to use bidirectional RNN for melody.
            dropout (float): Dropout rate for regularization.
            use_attention (bool): Whether to use attention mechanism. Default is False.
        """
        super(LyricsGenerator, self).__init__()
        self.use_attention = use_attention
        # Allow attention to only work for bi-directional architecture
        self.bidirectional_melody = True if use_attention else bidirectional_melody
        if (use_attention and not bidirectional_melody):
            print(f'[Warning]: The model implements attention mechanism only for bi-directional melody processing. __init__ changed automatically to bidirectional_melody=True, use_attention=True')
        
        # Extract vocab and Word2Vec from the dataset
        self.vocab = dataset.vocab
        self.vocab_inv = dataset.vocab_inv
        self.word_vocab_size = len(self.vocab)
        self.unk_index = dataset.vocab[UNKNOWN_TOKEN]

        # Extract pre-trained embeddings from the dataset
        embedding_weights = dataset.embedding_weights
        
        # Word Embedding Layer
        self.word_embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        
        # Melody RNN
        self.melody_rnn = nn.LSTM(
            input_size=melody_dim,
            hidden_size=hidden_dim,
            bidirectional=self.bidirectional_melody,
            batch_first=True
        )
        
        # Lyrics RNN
        self.lyrics_rnn = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Attention mechanism
        if use_attention:
            melody_dim = hidden_dim * (2 if self.bidirectional_melody else 1)
            self.attention_layer = Attention(
                query_dim=hidden_dim,        # Query dimension (melody context)
                key_dim=hidden_dim,          # Key dimension (lyrics output)
                value_dim=hidden_dim,        # Value dimension (lyrics output)
                output_dim=hidden_dim        # Output dimension
            )
            # Add projection for melody_context
            self.melody_context_projection = nn.Linear(melody_dim, hidden_dim)
            # -----------
            # Projection for combined context in generation
            # This ensures the concatenated context aligns with the fc input
            combined_context_dim = hidden_dim + (hidden_dim * 2 if self.bidirectional_melody else hidden_dim)
            self.combined_context_projection = nn.Linear(combined_context_dim, hidden_dim)
            # -----------
        
        # Dropout layer before FC
        self.fc_dropout = nn.Dropout(dropout)
        
        # Final Fully Connected Layer
        melody_dim = hidden_dim * (2 if self.bidirectional_melody else 1)
        fc_input_dim = hidden_dim if use_attention else (hidden_dim + melody_dim)
        self.fc = nn.Linear(fc_input_dim, self.word_vocab_size)


    def forward(self, word_input, melody_input):
        """
        Forward pass for the model.

        Args:
            word_input (torch.Tensor): Input tensor of word indices (batch_size, seq_len).
            melody_input (torch.Tensor): Input tensor of melody features (batch_size, melody_seq_len, melody_dim).

        Returns:
            torch.Tensor: Output logits for word predictions (batch_size, seq_len, vocab_size).
        """
        # Embedding the word input
        word_embedded = self.word_embedding(word_input)  # (batch_size, seq_len, word_embedding_dim)

        # Process the melody input through the melody RNN
        melody_output, (melody_hn, _) = self.melody_rnn(melody_input)

        # Handle bidirectional melody RNN output
        if self.bidirectional_melody:
            melody_context = torch.cat((melody_hn[-2], melody_hn[-1]), dim=-1)
        else:
            melody_context = melody_hn[-1]

        # Process the word embeddings through the lyrics RNN
        lyrics_output, _ = self.lyrics_rnn(word_embedded)

        if self.use_attention:
            # Project melody_context to match hidden_dim
            projected_melody_context = self.melody_context_projection(melody_context)
            
            # Use the attention layer (attention per token)
            context = self.attention_layer(
                query=lyrics_output,                          # Query: Lyrics RNN output
                keys=projected_melody_context.unsqueeze(1),   # Keys: Projected melody context
                values=projected_melody_context.unsqueeze(1)  # Values: Projected melody context
            )
            # Combine and project context
            expanded_melody_context = melody_context.unsqueeze(1).expand(-1, context.size(1), -1)
            combined_context = torch.cat((context, expanded_melody_context), dim=-1)
            context = self.combined_context_projection(combined_context)

        else:
            # Expand the melody context to match the sequence length
            expanded_melody_context = melody_context.unsqueeze(1).expand(-1, lyrics_output.size(1), -1)
            # Concatenate the lyrics RNN output and melody context
            context = torch.cat((lyrics_output, expanded_melody_context), dim=-1)

        # Apply dropout and pass through the fully connected layer
        context = self.fc_dropout(context)
        logits = self.fc(context)

        return logits

    
    def __calculate_penalty(self, outputs, special_token_indices, penalty_weight, device):
        """
        Hidden class method to calculate penalties for long lines, text length, and parentheses imbalance.
        Penalties are integrated within the model to direct it towards better text generations.
        """
        batch_size, seq_len, vocab_size = outputs.shape
        penalties = 0.0

        # Predicted tokens from the model's outputs
        predicted_tokens = outputs.argmax(dim=-1)  # Shape: (batch_size, seq_len)

        # Long line penalty
        newline_token_idx = self.vocab.get(NEWLINE_TOKEN, None)
        if newline_token_idx is not None:
            newline_counts = (predicted_tokens == newline_token_idx).sum(dim=1)  # Count NEWLINE_TOKEN per batch
            long_line_penalty = (newline_counts.float().mean() - 5).clamp(min=0)  # Penalize lines longer than 5
            penalties += long_line_penalty

        # Missing EOT_TOKEN & CHORUS_TOKEN penalties
        eot_token_idx = self.vocab.get(EOT_TOKEN, None)
        chorus_token_idx = self.vocab.get(CHORUS_TOKEN, None)
        if eot_token_idx is not None:
            eot_present = (predicted_tokens == eot_token_idx).any(dim=1)  # Check for EOT presence per batch
            eot_penalty = (~eot_present).float().mean()  # Penalize missing EOT_TOKEN
            penalties += eot_penalty
        if chorus_token_idx is not None:
            chorus_present = (predicted_tokens == chorus_token_idx).any(dim=1)  # Check for CHORUS presence per batch
            chorus_penalty = (~chorus_present).float().mean()  # Penalize missing CHORUS
            penalties += chorus_penalty

        # Parentheses imbalance penalty (within segments)
        open_token_idx = self.vocab.get("(", -1)
        close_token_idx = self.vocab.get(")", -1)
        if open_token_idx != -1 and close_token_idx != -1:
            unmatched_penalty = 0.0
            for batch in range(batch_size):
                stack = []
                for token in predicted_tokens[batch]:
                    if token == open_token_idx:
                        stack.append(token)  # Push '(' onto the stack
                    elif token == close_token_idx:
                        if stack:
                            stack.pop()  # Match '(' with ')'
                        else:
                            unmatched_penalty += 1  # Unmatched ')'
                unmatched_penalty += len(stack)  # Penalize unmatched '(' left in the stack
            penalties += unmatched_penalty / batch_size  # Average penalty over the batch

        # Final penalty weighted by penalty_weight
        return penalty_weight * penalties


    def train_model(self, dataset, batch_size, chunk_size, stride, lr, weight_decay, teacher_forcing_ratio, penalty_weight, epochs, device, 
                    val_ratio, stopping_criteria=3, shuffle=True, log_dir="/content/runs", checkpoint_path="Trained Models/best_model.pth"):
        """
        Train the model with teacher forcing, validation, and log metrics to TensorBoard.
        The training process enforces generation guidelined within the loss function calculation for each batch.

        Args:
            dataset (LyricsDataset): The dataset to use for training and validation.
            batch_size (int): Batch size for DataLoaders.
            chunk_size (int): Size of each chunk in ChunkedLyricsDataset.
            stride (int): Stride for creating overlapping chunks.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer regularization.
            teacher_forcing_ratio (float): Probability of using teacher forcing at each time step.
            epochs (int): Number of epochs to train.
            device (torch.device): Device to use for training (CPU or GPU).
            val_ratio (float): Proportion of dataset to use for validation. Default is 0.2 (20%).
            stopping_criteria (int): Number of epochs with no improvement to wait until early stopping.
            shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.
            log_dir (str): Directory for TensorBoard logs. Default is "runs".
            checkpoint_path (str): Path to save the best model checkpoint. Default is "best_model.pth".

        Returns:
            dict: Dictionary containing training and validation losses for each epoch.
        """
        # Create DataLoaders
        train_loader, val_loader = get_data_loader(
            base_dataset=dataset,
            batch_size=batch_size,
            chunk_size=chunk_size,
            stride=stride,
            val_ratio=val_ratio,
            shuffle=shuffle
        )

        # Optimizer and Criterion
        special_token_indices = {self.vocab[token] for token in SPECIAL_TOKENS if token in self.vocab}
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.02)

        # TensorBoard setup
        writer = SummaryWriter(log_dir=log_dir)
        self.to(device)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        best_val_loss = float('inf')
        best_model_state = None
        start_time = time.time()
        epochs_no_improvement = 0

        for epoch in range(1, epochs + 1):
            # Reduce teachers forcing as a function of epoch
            teacher_forcing_ratio *= (1 - epoch/epochs)             
            
            # Training loop
            self.train()
            train_loss = 0.0
            for word_input, melody_input, targets in train_loader:
                word_input, melody_input, targets = word_input.to(device), melody_input.to(device), targets.to(device)
                optimizer.zero_grad()

                batch_size, seq_len = word_input.size()
                outputs = torch.zeros(batch_size, seq_len, self.word_vocab_size, device=device)

                # Unroll sequence with teacher forcing logic
                hidden_state = None
                for t in range(seq_len):
                    current_input = word_input[:, t].unsqueeze(1)  # (batch_size, 1)
                    output = self(current_input, melody_input)  # Forward pass
                    outputs[:, t, :] = output.squeeze(1)

                    # Determine whether to use teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        next_input = word_input[:, t].unsqueeze(1)  # Use ground truth
                    else:
                        next_input = output.argmax(dim=-1).unsqueeze(1)  # Use model's prediction
                    current_input = next_input
                
                # Compute penalties on loss function
                total_penalty = self.__calculate_penalty(outputs=outputs, 
                                                        special_token_indices=SPECIAL_TOKENS, 
                                                        penalty_weight=penalty_weight, 
                                                        device=device)

                # Compute loss and add penalties
                base_loss = criterion(outputs.view(-1, self.word_vocab_size), targets.view(-1))
                total_loss = base_loss + total_penalty
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

            train_loss /= len(train_loader)

            # Validation loop
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (word_input, melody_input, targets) in enumerate(val_loader):
                    word_input, melody_input, targets = word_input.to(device), melody_input.to(device), targets.to(device)
                    outputs = self(word_input, melody_input)
                    # Compute penalties on loss function
                    total_penalty = self.__calculate_penalty(outputs=outputs, 
                                                            special_token_indices=SPECIAL_TOKENS, 
                                                            penalty_weight=penalty_weight, 
                                                            device=device)
                    base_loss = criterion(outputs.view(-1, self.word_vocab_size), targets.view(-1))
                    total_loss = base_loss + total_penalty
                    val_loss += total_loss.item()

            val_loss /= len(val_loader)
            
            # Step the scheduler based on small reduction in val_loss
            scheduler.step(val_loss)

            # Check for the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                torch.save(best_model_state, checkpoint_path)
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement == stopping_criteria:
                    break

            # TensorBoard logging
            writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('Teacher Forcing Ratio', teacher_forcing_ratio, epoch)

            training_time = round((time.time() - start_time)/60,2)
            print(f"[Training Status]: Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]}, Teacher's Ratio: {teacher_forcing_ratio:.3f}, Time: {training_time:.2f}m")

        writer.close()

        # Load the best model
        self.load_state_dict(best_model_state)
        print(f"[Training Status]: Best model with Validation Loss: {best_val_loss:.4f} saved to {checkpoint_path}")

        return self


    def generate_text(self, song_name, max_length, device, start_token, temperature=1.0, test_csv='Data/lyrics_test_set.csv'):
        """
        Generate text based on a given song's melody and a starting token.

        Args:
            song_name (str): Name of the song to generate lyrics for.
            max_length (int): Maximum number of tokens to generate.
            device (torch.device): Device for generation (CPU or GPU).
            start_token (str): The start token for the generation. Should default to <BeginningOfText>.
            temperature (float): Sampling temperature for randomness.
            test_csv (str): Path to the test dataset CSV.

        Returns:
            str: Generated text.
        """
        self.to(device)
        self.eval()

        # Load and preprocess test dataset
        test_df = pd.read_csv(test_csv, header=None, names=["Artist", "Song Name", "Lyrics"])
        test_df["Artist"] = test_df["Artist"].str.strip().str.replace(r'\s+', ' ', regex=True)
        test_df["Song Name"] = test_df["Song Name"].str.strip().str.replace(r'\s+', ' ', regex=True)
        song_row = test_df[test_df["Song Name"].str.lower() == song_name.lower()]
        if song_row.empty:
            raise ValueError(f"Song '{song_name}' not found in the test dataset.")

        processed_lyrics = song_row["Lyrics"].iloc[0]
        artist = song_row["Artist"].iloc[0]
        processed_lyrics, _ = process_lyrics(processed_lyrics, artist)

        # Generate melody input for the song
        midi_file = os.path.join("Data/midi_files", f"{artist} - {song_name}.mid".replace(" ", "_"))
        if not os.path.exists(midi_file):
            raise ValueError(f"MIDI file for song '{song_name}' not found.")

        midi = pretty_midi.PrettyMIDI(midi_file)
        melody_input = torch.tensor(
            process_midi(midi, processed_lyrics), dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Prepare for generation
        start_token = self.vocab.get(start_token, self.unk_index)
        melody_output, (melody_hn, _) = self.melody_rnn(melody_input)

        if self.bidirectional_melody:
            melody_context = torch.cat((melody_hn[-2], melody_hn[-1]), dim=-1)
        else:
            melody_context = melody_hn[-1]

        generated_tokens = []
        current_token = torch.tensor([[start_token]], device=device)

        for _ in range(max_length):
            word_embedded = self.word_embedding(current_token)
            lyrics_output, _ = self.lyrics_rnn(word_embedded)

            if self.use_attention:
                # Project melody_context to match hidden_dim
                projected_melody_context = self.melody_context_projection(melody_context)
                expanded_melody_context = projected_melody_context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
                # Use attention layer
                attention_context = self.attention_layer(
                    query=lyrics_output,            # Query: Lyrics output
                    keys=expanded_melody_context ,  # Keys: Projected melody context
                    values=expanded_melody_context  # Values: Projected melody context
                )
                combined_context = torch.cat((attention_context.squeeze(1), melody_context), dim=-1)
                combined_context = self.combined_context_projection(combined_context)

            else:
                combined_context = torch.cat((lyrics_output[:, -1, :], melody_context), dim=-1)
            
            logits = self.fc(combined_context)
            probs = F.softmax(logits / temperature, dim=-1).squeeze()

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == self.vocab.get(EOT_TOKEN):
                break  # Stop generation at EOT token

            generated_tokens.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)

        # Map indices back to words and handle replacements
        start_word = self.vocab_inv.get(start_token, UNKNOWN_TOKEN)
        generated_text = [start_word]
        for token in generated_tokens:
            word = self.vocab_inv.get(token, None)
            if word == NEWLINE_TOKEN:
                generated_text.append("\n")
            elif word == CHORUS_TOKEN and CHORUS_TOKEN not in generated_text: # Allow 1 CHORUS
                generated_text.append("\n\nChorus:\n")
            elif word not in [BOT_TOKEN, EOT_TOKEN, UNKNOWN_TOKEN]:
                generated_text.append(word)
            elif not word:
                continue
        print(f'[Info]: Generated sequence length: {len(generated_text)} tokens.')
        
        # Prettify the text a little
        raw_text = ' '.join(generated_text)
        processed_text = re.sub(r'\s([?!.,;:])', r'\1', raw_text)   # Remove spaces before punctuation marks
        processed_text = re.sub(r'([\(\[{])\s+', r'\1', processed_text)  # No space after opening brackets
        processed_text = re.sub(r'\s+([\)\]}])', r'\1', processed_text)  # No space before closing brackets
        
        return processed_text
