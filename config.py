# ---- Data Settings ---- #
data_path = 'dataset/BCIC_2a/'  # Path to the EEG dataset
train_files = ['training.mat']            # Training dataset file name
test_files = ['evaluation.mat']           # Testing dataset file name
output = 'output'                         # Directory to save outputs (models, logs, etc.)
model_name = "CFA"                       # The name of model to save
batch_size = 24                           # Batch size for training and testing
num_segs = 8                              # Number of segments for data augmentation

# ---- Model Settings ---- #
#NULL

# ---- Training Settings ---- #
epochs = 2000                             # Number of training epochs
lr = 2 ** -12                             # Learning rate
weight_decay = 1e-4                       # Weight decay for optimizer
