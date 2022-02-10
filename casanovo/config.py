"""
Data, model and training/testing related options can be specified here.
"""
import os

#Random seed to be used across the pipeline
random_seed = 454

#Train data options
train_annot_spec_idx_path = os.path.join(os.getcwd(),'casanovo_train.hdf5') #path to write the training data index file
train_spec_idx_overwrite = True

#Validation data options
val_annot_spec_idx_path = os.path.join(os.getcwd(),'casanovo_val.hdf5') #path to write the validation data index file
val_spec_idx_overwrite = True

#Test data options
test_annot_spec_idx_path = os.path.join(os.getcwd(),'casanovo_test.hdf5') #path to write the test data index file
test_spec_idx_overwrite = True

#Preprocessing parameters
preprocess_spec = False
n_peaks = 150
min_mz = 1.0005079* 50.5

#Hardware options
num_workers = 0
gpus = None #None for CPU, int list to specify GPUs

#Model options
max_charge = 10
dim_model = 512
n_head = 8
dim_feedforward = 1024
n_layers = 9
dropout = 0
dim_intensity = None
custom_encoder = None
max_length = 100
residues = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        # AA mods:
        "M+15.995": 131.040484645 + 15.994915,  # Met Oxidation
        "N+0.984": 114.042927470 + 0.984016,  # Asn Deamidation
        "Q+0.984": 128.058577540 + 0.984016,  # Gln Deamidation
    }
max_charge = 10
n_log = 1
tb_summarywriter = None
warmup_iters = 100000
max_iters = 600000
learning_rate = 5e-4
weight_decay = 1e-5
reverse_peptide_seqs = True

#Training/inference options
train_batch_size = 32
val_batch_size = 1024
test_batch_size = 1024

accelerator="ddp"
logger = None
max_epochs = 30
num_sanity_val_steps = 0

train_from_scratch = True

save_model = False
model_save_folder_path = ''
save_weights_only = True
every_n_epochs = 1
