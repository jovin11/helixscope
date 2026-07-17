# Model card: HelixScope CNN–LSTM Q3 predictor

## Code map

- `python_service/pssp_model/model.py` defines the CNN-LSTM architecture used
  when creating a new model.
- `python_service/pssp_model/inference.py` loads the saved Keras artifact lazily
  and performs validation, encoding, prediction, and decoding.
- `python_service/pssp_model/training.py` provides explicit `train_model` and
  `fine_tune_model` entry points for prepared NumPy datasets. Training and PDB
  downloads are intentionally excluded from CI.
- `python_service/pssp_model/preprocessing.py` is the shared encoding contract
  for training and serving.

The Flask app constructs `ProteinStructurePredictor` once and calls it from the
existing `/predict` and `/api/v1/predict` routes. The checkpoint remains
lazy-loaded until the first prediction request.

## Model details

| Field | Value |
|---|---|
| Task | Residue-level protein secondary structure prediction |
| Output space | Q3: helix (`H`), sheet (`E`), coil (`C`) |
| Input | Up to 400 canonical amino-acid residues |
| Encoding | 20-channel one-hot representation, zero-padded to 400 |
| Architecture | 4 Conv1D blocks, batch normalization, 3 LSTMs, time-distributed classifier |
| Framework | TensorFlow / Keras |
| Current artifact | `best_model_v2.keras` |
| Artifact SHA-256 | `2cc08c588cb072880b3a8782393a32758b5a4f178fbbd80bee7b13eec32ec683` |
| Artifact saved | 2024-08-29 with Keras 3.5.0 |

## Intended use

The model is intended for education, software demonstration, and exploratory
research. It can help illustrate how sequence models map amino-acid context to
coarse secondary-structure hypotheses.

It is not intended for clinical decisions, safety-critical workflows, definitive
structural annotation, drug development decisions, or use without comparison to
validated bioinformatics tools and experimental evidence.

## Training data and labels

The original pipeline downloads and parses Protein Data Bank records. It extracts
protein sequences from `SEQRES`, maps residue indices using `DBREF` and `SEQADV`,
and obtains Q3 labels from PDB `HELIX` and `SHEET` annotations. Residues not covered
by those records are labeled coil.

The repository retains a snapshot of 2,234 `.ent` files. It does not currently
retain a machine-readable manifest that identifies the exact training and
validation membership for the saved model.

## Evaluation

Original experiment comments record validation accuracy of approximately 0.72–0.73.
This is a historical development result, not a publication-quality benchmark.
The following artifacts are missing and should be recreated before comparing this
model with other methods:

- deterministic, accession-level split manifests;
- sequence-similarity or homology controls between splits;
- explicit masking of zero-padded positions in metrics and loss;
- per-class precision, recall, and F1;
- segment overlap (SOV) evaluation;
- calibration and external-dataset evaluation.

## Limitations

- Only the 20 canonical amino-acid symbols are accepted.
- Sequences longer than 400 residues are rejected rather than windowed.
- Q3 collapses richer DSSP-style structural states into three broad classes.
- PDB header annotations may differ from DSSP-derived labels.
- The unidirectional LSTM architecture may underuse future residue context.
- Confidence is the largest softmax output and is not calibrated probability.
- Similar proteins may occur across the original split; homology leakage was not
  measured.
- The training distribution inherits selection biases from deposited PDB
  structures and the project's PDB sampling procedure.

## Recommendations

Treat outputs as hypotheses. For research use, compare predictions with an
established secondary-structure predictor, inspect low-confidence regions, and
validate conclusions against experimental structures when available. Any future
model should be evaluated on a frozen, homology-aware test set with padding-aware
metrics before release.
