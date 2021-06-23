# All required config values are here
target_vocab_size = 2**15
max_gradient_norm = 5
BUFFER_SIZE = 35000
BATCH_SIZE = 5 # try bigger batch for faster training
EPOCHS = 5
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
dia_of_embedding = 128
units = 256  # from pointer generator paper,
text_max_len = 128
sum_max_len = 50