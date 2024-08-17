import os
import sys
import contextlib
import numpy as np
import tensorflow as tf
from Bio.SeqUtils import seq1
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed, MaxPooling1D
from tensorflow.keras.layers import Input, Embedding, GlobalMaxPooling1D, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from Bio.PDB import PDBList, PDBParser, DSSP

# ------------------------------------------------------------------------------------------------------------------------------------
# List of more PDB IDs for a larger dataset (example IDs, replace with actual ones)
pdb_ids = ['1A8M', '1BNA', '1C0A', '1D3Z', '1E7P', '1F9T', '1GID', '1H9N', '1JFF', '1KX5', '1L2Y', '1N8O', '1O5R', '1P6X', '1R4Z', '1S7Y', '1T8W', '1U9Z', '1A6M', '1BVC', '1CEX', '1D4T', '1E7O', '1F3H', '1GFL', '1HNI', '1IDH', '1J5E', '1K5D', '1M1T', '1N9X', '1O3Y', '1Q8Z', '1R2Y', '1S1D', '1U6Q', '1V3W', '1W1Z', '1X3A', '1Y4B', '1Z6C', '2A7D', '2B8E', '2C9F', '2D0G', '2E1H', '2F2I', '2G3J', '2H4K', '2I5L', '2J6M', '2K7N', '2M9P', '2N0Q', '2P2S', '2Q3T', '2R4U', '2V8Y', '2W9Z', '3A1A', '3C3C', '3D4D', '3E5E', '3F6F', '3G7G', '3H8H', '3I9I', '3J0J', '3K1K', '3L2L', '3M3M', '3N4N', '3O5O', '3P6P', '3Q7Q', '3R8R', '3S9S', '3T0T', '3U1U', '3V2V', '3W3W', '4B2B', '4C3C', '4D4D', '4E5E', '4F6F', '4G7G', '4H8H', '4I9I', '4J0J', '4K1K', '4L2L', '4M3M', '4N4N', '4O5O', '4P6P', '4Q7Q', '4R8R', '4V2V', '4X4X', '4Y5Y', '4Z6Z', '5A1A', '5C3C', '5D4D', '5E5E', '5F6F', '5H8H', '5I9I', '5J0J', '5K1K', '2HRK', '3A5X', '4XYZ', '6ZP7', '3R2C', '2FL7', '4H57', '1BCX', '2B4D', '5C67']

# # Test for duplicates
def duplicates(pdb):
    tmp = []
    for id in pdb:
        if id in tmp:
            print(id)
        else:
            tmp.append(id)
    return tmp

def count_list(pdb):
    count = 0
    for id in pdb:
        count += 1
    print("Count of list is: " + str(count))

def remove_invalid(pdb, e_list):
    tmp = []
    for id in pdb:
        if id not in e_list:
            tmp.append(id)
    print(tmp)

# remove_invalid(pdb_ids, ['4A1A'])
# count_list(pdb_ids)

# Doesn't exist: 1p6z, 1t5v, 2o1r, 2s5v, 2t6w, 2u7x, 3b2b
# Discrepancy list: The following has some error, check them out later to see if the model isn't accurate enough, but overall it works well
# 1p6x, 1t8w, 1q8z, 2b8e, 2c9f, 2l8o, 2p2s, 2w9z, 3a1a, 3k1k, 3l2l, 3m3m, 3o5o, 3u1u, 3w3w, 4b2b, 4e5e, 4q7q, 4y5y, 5h8h, 3r2c, 2b4d, 5c67
# ------------------------------------------------------------------------------------------------------------------------------------

# Create directory for PDB files
pdb_dir = 'pdb_files'
os.makedirs(pdb_dir, exist_ok=True)

# Function to suppress output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Function to download PDB files
def download_pdb(pdb_id, pdb_dir='pdb_files'):
    pdbl = PDBList()
    with suppress_output():
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')

# ------------------------------------------------------------------------------------------------------------------------------------

# seq1 import from biopython has inbuilt function which serves the same purpose
# # Function to convert three-letter codes to single-letter codes
# def three_to_one(three_letter_code):
#     three_to_one_dict = {
#         'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
#         'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
#         'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
#         'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
#     }
#     return three_to_one_dict.get(three_letter_code, 'X')  # 'X' for unknown residues


# Function to parse PDB files and extract sequences and secondary structures
def parse_pdb_multiple_chains(file_path):
    sequences = {}
    secondary_structures = {}
    residue_to_index = {}
    chain_start_residues = {}
    chain_end_residues = {}
    modres_mapping = {}
    decrepency_list = []

    def three_to_one(residue):
        try:
            return seq1(residue)
        except:
            return 'X'

    def initialize_secondary_structure(chain_id, length):
        secondary_structures[chain_id] = ['C'] * length

    def is_nucleotide_sequence(sequence):
        # List of 20 standard amino acids
        amino_acids = {
            'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
            'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
            'MET', 'ASN', 'PRO', 'GLN', 'ARG',
            'SER', 'THR', 'VAL', 'TRP', 'TYR'
        }
        return not all(residue in amino_acids for residue in sequence)

    with open(file_path, 'r') as file:
        # print(file_path)
        for line in file:
            if line.startswith('DBREF'):
                chain_id = line[12]
                start_residue = int(line[14:18].strip())
                end_residue = int(line[20:24].strip())
                chain_start_residues[chain_id] = start_residue
                chain_end_residues[chain_id] = end_residue
            if line.startswith('MODRES'):
                modres = line[12:15].strip()
                original_residue = line[24:27].strip()
                modres_mapping[modres] = original_residue

        file.seek(0)  # Reset file pointer to the beginning
        for line in file:
            if line.startswith('SEQRES'):
                chain_id = line[11]
                if chain_id not in sequences:
                    sequences[chain_id] = []
                    residue_to_index[chain_id] = {}
                parts = line.split()
                residues = parts[4:]
                if is_nucleotide_sequence(residues):
                    # print(f"Nucleotide or non-protein sequence found in chain {chain_id}, skipping...")
                    continue
                residues = [three_to_one(res) for res in residues]
                seq_position = len(sequences[chain_id])
                start_residue_number = chain_start_residues.get(chain_id, 1)
                end_residue_number = chain_end_residues.get(chain_id, start_residue_number + len(residues) - 1)
                for i in range(len(residues)):
                    residue_number = start_residue_number + seq_position + i
                    residue_to_index[chain_id][residue_number] = seq_position + i
                sequences[chain_id].extend(residues)

                # print(f"SEQRES: chain={chain_id}, residues={residues}")
            elif line.startswith('HELIX') or line.startswith('SHEET'):
                chain_id = line[19] if line.startswith('HELIX') else line[21]
                if chain_id not in sequences:
                    continue
                try:
                    start_residue = int(line[21:25].strip()) if line.startswith('HELIX') else int(line[22:26].strip())
                    end_residue = int(line[33:37].strip())
                    sec_type = 'H' if line.startswith('HELIX') else 'E'
                    # print(f"{'HELIX' if line.startswith('HELIX') else 'SHEET'}: chain={chain_id}, start={start_residue}, end={end_residue}")

                    # Initialize secondary structure if not already done
                    if chain_id not in secondary_structures:
                        initialize_secondary_structure(chain_id, len(sequences[chain_id]))

                    # Debug print to show the mapping of residues
                    # print(f"Residue to index mapping for chain {chain_id}: {residue_to_index[chain_id]}")

                    # Update secondary structure
                    for residue_num in range(start_residue, end_residue + 1):
                        if residue_num in residue_to_index[chain_id]:
                            index = residue_to_index[chain_id][residue_num]
                            secondary_structures[chain_id][index] = sec_type
                            # print(f"Updating index {index} for chain {chain_id} to {sec_type}")
                        else:
                            # print(f"Residue number {residue_num} not found in mapping for chain {chain_id}")
                            if file_path not in decrepency_list:
                                decrepency_list.append(file_path) 
                except ValueError as e:
                    print(f"Error parsing {'HELIX' if line.startswith('HELIX') else 'SHEET'} line: {line}")

    for chain_id in sequences:
        sequence_length = len(sequences[chain_id])
        if chain_id not in secondary_structures:
            initialize_secondary_structure(chain_id, sequence_length)
        secondary_structures[chain_id] = secondary_structures[chain_id][:sequence_length]

    for chain_id in sequences:
        sequences[chain_id] = ''.join(sequences[chain_id])
        secondary_structures[chain_id] = ''.join(secondary_structures[chain_id])

    return sequences, secondary_structures, decrepency_list

# # Test the function
# 1p6x, 1t8w, 1q8z, 2b8e, 2c9f, 2l8o, 2p2s, 2w9z, 3a1a, 3k1k, 3l2l, 3m3m, 3o5o, 3u1u, 3w3w, 4b2b, 4e5e, 4q7q, 4y5y, 5h8h, 3r2c, 2b4d, 5c67
# sequences, secondary_structures = parse_pdb_multiple_chains("pdb_files/pdb3m3m.ent")
# print(sequences)
# print(secondary_structures)


# 1a8m, 1bna, 1c0a, 1d3z, 1e7p, 1f9t, 1gid, 1h9n, 1jff, 1kx5
# # Example usage
# file_path = 'pdb_files/pdb1f9t.ent'
# sequences, secondary_structures = parse_pdb_multiple_chains(file_path)
# for chain_id in sequences:
#     print(f'Chain {chain_id}:')
#     print('Sequence:', sequences[chain_id])
#     print('Secondary Structure:', secondary_structures[chain_id])
# print(sequences)

# # Verify if the secondary structure annotations match the expected HELIX and SHEET records
# for chain_id in sequences:
#     print(f'Chain {chain_id} Length: {len(sequences[chain_id])}')
#     print(f'Secondary Structure Length: {len(secondary_structures[chain_id])}')
#     assert len(sequences[chain_id]) == len(secondary_structures[chain_id]), "Length mismatch between sequence and secondary structure"
#     print(f'First 100 residues of Chain {chain_id}: {sequences[chain_id][:100]}')
#     print(f'First 100 annotations of Secondary Structure of Chain {chain_id}: {secondary_structures[chain_id][:100]}')

# ------------------------------------------------------------------------------------------------------------------------------------

# Function to preprocess sequences (primary)
def preprocess_sequences(sequence, max_length=700):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoding = np.zeros((max_length, len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        if aa in aa_dict:
            encoding[i, aa_dict[aa]] = 1
    return encoding

## Testing the function
# print(preprocess_sequences('ACDEFGHIKLMNPQRSTVWYAAAA', 24))

# Function to decode one-hot encoded sequences
def decode_sequence(one_hot_matrix):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''
    for row in one_hot_matrix:
        index = np.argmax(row)
        if np.sum(row) > 0:  # Ensure there's a non-zero row
            sequence += amino_acids[index]
        else:
            break  # Stop decoding when you encounter the first all-zero row
    return sequence

# Decode the first sequence
# decoded_sequence_1A8M = decode_sequence(X[0])
# print(f"Decoded sequence for '1A8M': {decoded_sequence_1A8M}")
# print(f"First 10 residues of '1A8M': {decoded_sequence_1A8M[:10]}")
# print(f"Length of the sequence for '1A8M': {len(decoded_sequence_1A8M)}")

# one_hot_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

# # Decode the sequence
# decoded_sequence = decode_sequence(one_hot_matrix)
# print(decoded_sequence)  
# #VRSSS

# # Example: Print the first sequence to inspect
# print("First sequence (X[0]):")
# print(X[0])

# Example: Print a specific part of the first sequence to inspect
# print("First 5 residues of the first sequence (X[0][:5]):")
# print(X[0][:5])

# ------------------------------------------------------------------------------------------------------------------------------------

# Function to preprocess secondary structures (secondary sequences)
def preprocess_structures(structure, max_length=700):
    structure_dict = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
    encoding = np.zeros((max_length, 3), dtype=np.float32)
    for i, s in enumerate(structure[:max_length]):
        if s in structure_dict:
            encoding[i, structure_dict[s]] = 1
    return encoding


## Tests for secondary structure encoding and decoding

# struct = 'HHHCCEECHEHCHCCCC'
# print(preprocess_structures(struct, 17))

# # Function to decode the encoded secondary structures back to the string representation
# def decode_structures(encoding):
#     index_to_structure = {0: 'H', 1: 'E', 2: 'C'}
#     structure = []
#     for row in encoding:
#         index = np.argmax(row)
#         structure.append(index_to_structure.get(index, 'C'))
#     return ''.join(structure)

# # Test the encoding and decoding process
# def test_preprocess_and_decode():
#     # Define some test cases
#     test_cases = [
#         "HHHEEECCC",
#         "CCHHHEECCCCHHHEE",
#         "ECCCCCHHHHEEE",
#         "HHCCHHEECC",
#         "HHHHHHHHHH",
#         "EEEEEEEEEE",
#         "CCCCCCCCCC",
#         "HHHEECCCHHEECC"
#     ]
    
#     # Maximum length for encoding
#     max_length = 18

#     # Run the tests
#     for case in test_cases:
#         print(f"Original structure: {case}")
#         encoded = preprocess_structures(case, len(case))
#         print("Encoded structure: ")
#         print(f"{encoded}")
#         decoded = decode_structures(encoded[:len(case)])  # Only decode the part of the array that was used
#         print(f"Decoded structure:  {decoded}")
#         assert case == decoded, f"Test failed for case: {case}"
#         print("Test passed!\n")

# # Run the test function
# test_preprocess_and_decode()

# ------------------------------------------------------------------------------------------------------------------------------------
# change max length to 450, 500
# Function to build the dataset
def build_dataset(pdb_ids, pdb_dir='pdb_files', max_length=400):
    X, y = [], []
    error_list = []
    decrepency_list = []
    for pdb_id in pdb_ids:
        try:
            download_pdb(pdb_id, pdb_dir)
            file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
            sequences, secondary_structures, d_list = parse_pdb_multiple_chains(file_path)
            if d_list != []:
                decrepency_list.append(d_list[0])
            for chain_id in sequences:
                X.append(preprocess_sequences(sequences[chain_id], max_length))
                y.append(preprocess_structures(secondary_structures[chain_id], max_length))
        except Exception as e:
            print(f'Error processing {pdb_id}: {e}')
            error_list.append(pdb_id)
    return np.array(X), np.array(y), error_list, decrepency_list

# Build the dataset
X, y, error_list, decrepency_list = build_dataset(pdb_ids)
# print(error_list)
# print(decrepency_list)
print(f'Dataset shape: X={X.shape}, y={y.shape}')



# ------------------------------------------------------------------------------------------------------------------------------------

# function to analyze sequence lengths with graph representation using matplotlib
def analyze_sequence_lengths(pdb_ids, pdb_dir='pdb_files'):
    lengths = []
    for pdb_id in pdb_ids:
        try:
            file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
            sequences, _ = parse_pdb_multiple_chains(file_path)
            for chain_id in sequences:
                lengths.append(len(sequences[chain_id]))
        except Exception as e:
            print(f'Error processing {pdb_id}: {e}')
    return lengths

# # Analyze lengths
# sequence_lengths = analyze_sequence_lengths(pdb_ids)

# # Plot the distribution of sequence lengths
# plt.hist(sequence_lengths, bins=20)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.title('Distribution of Protein Sequence Lengths')
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------


# Split the data into training and validation sets
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Define the CNN-LSTM model
# model = Sequential([
#     Input(shape=(X.shape[1], X.shape[2])),
#     Masking(mask_value=0.0),
#     Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
#     MaxPooling1D(pool_size=1),
#     LSTM(128, return_sequences=True),
#     Dropout(0.5),  # Add dropout for regularization
#     Dense(3, activation='softmax')
# ])
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=1),
    LSTM(128, return_sequences=True),
    Dropout(0.5),  # Add dropout for regularization
    Dense(3, activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Add model checkpoint to save the best model in Keras format
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Load the saved model from the file
loaded_model = keras.models.load_model('best_model.keras')

# Example prediction using the loaded model
predictions = loaded_model.predict(X_val)
