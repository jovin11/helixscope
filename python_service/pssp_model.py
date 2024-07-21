import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import Input, Embedding, GlobalMaxPooling1D
from Bio.PDB import PDBList, PDBParser, DSSP
import sys
import contextlib
import matplotlib.pyplot as plt
from Bio.SeqUtils import seq1

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
        for line in file:
            if line.startswith('DBREF'):
                chain_id = line[12]
                start_residue = int(line[14:18].strip())
                chain_start_residues[chain_id] = start_residue

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
                for i in range(len(residues)):
                    residue_number = chain_start_residues.get(chain_id, 1) + seq_position + i  # Update residue number correctly
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
                            print(f"Residue number {residue_num} not found in mapping for chain {chain_id}")
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

    return sequences, secondary_structures

# # Test the function
# sequences, secondary_structures = parse_pdb_multiple_chains("pdb_files/pdb1kx5.ent")
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

# Function to build the dataset
def build_dataset(pdb_ids, pdb_dir='pdb_files', max_length=400):
    X, y = [], []
    for pdb_id in pdb_ids:
        try:
            download_pdb(pdb_id, pdb_dir)
            file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
            sequences, secondary_structures = parse_pdb_multiple_chains(file_path)
            for chain_id in sequences:
                X.append(preprocess_sequences(sequences[chain_id], max_length))
                y.append(preprocess_structures(secondary_structures[chain_id], max_length))
        except Exception as e:
            print(f'Error processing {pdb_id}: {e}')
    return np.array(X), np.array(y)


# Example PDB IDs for initial testing (you can replace these with any 10 PDB IDs)
pdb_ids = ['1A8M', '1BNA', '1C0A', '1D3Z', '1E7P', '1F9T', '1GID', '1H9N', '1JFF', '1KX5']

# Build the dataset
X, y = build_dataset(pdb_ids)
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

# # Define the CNN-LSTM model
# Define the model
def build_model(input_length, vocab_size, embedding_dim, max_length):
    primary_input = Input(shape=(input_length,), name='primary_input')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(primary_input)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)

    secondary_input = Input(shape=(max_length, 3), name='secondary_input')
    y = Conv1D(filters=128, kernel_size=5, activation='relu')(secondary_input)
    y = GlobalMaxPooling1D()(y)

    combined = tf.keras.layers.concatenate([x, y])

    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    output = Dense(3, activation='softmax')(z)

    model = Model(inputs=[primary_input, secondary_input], outputs=output)
    return model

# Compile the model
model = build_model(input_length=700, vocab_size=21, embedding_dim=50, max_length=700)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Assume X_primary and X_secondary are your input data and y are the labels
# X_primary = ...  # Primary sequences data
# X_secondary = ...  # Secondary structure data
# y = ...  # Labels

# Train the model
# model.fit([X_primary, X_secondary], y, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
# model.evaluate([X_primary, X_secondary], y_test)
