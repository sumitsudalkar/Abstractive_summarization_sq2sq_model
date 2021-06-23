import os
import random
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import sequence
import seq2seq as s2s
from textclean import clean_text
import configa as co

# Gpu setup are optional
def set_GPU():
    tf.keras.backend.clear_session() #- for easy reset of notebook state
    # chck if GPU can be seen by TF
    tf.config.list_physical_devices('GPU')
    #tf.debugging.set_log_device_placement(True)  # only to check GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("gigaword32k.enc")
# add start and end of sentence tokens
start = tokenizer.vocab_size + 1 
end = tokenizer.vocab_size
print(start, end)

# Setup tokenization methods
def encode(Text, summary, start=start, end=end, text_max_len=128, sum_max_len=50):
    tokens = tokenizer.encode(Text.numpy())
    if len(tokens) > text_max_len:
        tokens = tokens[:text_max_len]
    tokens = tokens
    pad = sequence.pad_sequences([tokens],padding='post',maxlen=text_max_len).squeeze() # it is remove single-dimensional entries
    text_encode = pad 

    tokens = [start] + tokenizer.encode(summary.numpy())
    if len(tokens) > sum_max_len:
        tokens = tokens[:sum_max_len]
    tokens = tokens + [end]
    pad = sequence.pad_sequences([tokens], padding='post', 
                                maxlen=sum_max_len).squeeze()
    sum_encode = pad 

    return text_encode, sum_encode

def tf_encode(Text, summary):
    text_encode, sum_encode = tf.py_function(encode, [Text, summary], 
                                    [tf.int64, tf.int64])
    text_encode.set_shape([None])
    sum_encode.set_shape([None])
    return text_encode, sum_encode

#Setup model
BATCH_SIZE = 1  # for inference
dia_of_embedding = co.dia_of_embedding
units = co.units # from pointer generator paper
vocab_size = end + 2

# Create encoder and decoder objects
encoder = s2s.Encoder(vocab_size, dia_of_embedding, units, BATCH_SIZE)
decoder = s2s.Decoder(vocab_size, dia_of_embedding, units, BATCH_SIZE)

for layer in decoder.layers:
    print(layer.name)
    
for layer in encoder.layers:
    print(layer.name)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# Hydrate the model from saved checkpoint
checkpoint_dir =  './training_checkpoints-2021-May-02-01-28-02'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder,
                                decoder=decoder)

# The last training checkpoint
tf.train.latest_checkpoint(checkpoint_dir)
chkpt_status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
chkpt_status.assert_existing_objects_matched()


text_max_len = co.text_max_len
sum_max_len = co.sum_max_len

def greedy_search(Text):
    tokens = tokenizer.encode(Text) 
    if len(tokens) > text_max_len:
        tokens = tokens[:text_max_len]

    inputs = sequence.pad_sequences([tokens],padding='post',
                                maxlen=text_max_len).squeeze()
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)
    
    # output summary tokens will be stored in this
    summary = ' '
    hidden = [tf.zeros((1, units)) for i in range(2)] #BiRNN
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start], 0)
    for t in range(sum_max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                        dec_hidden,
                                                        enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if predicted_id == end:
            return summary, Text
        attention_weights = tf.reshape(attention_weights, (-1, ))
        summary += tokenizer.decode([predicted_id])
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id],0)
    return summary, Text

# Summarize
def summarize(Text, algo='greedy'):
    if algo == 'greedy':
        summary, Text = greedy_search(Text)
    else:
        print("Algorithm {} not implemented".format(algo))
        return
    print('Original Text: %s' % (Text),'\n')
    print('Summary: {}'.format(summary))
    

# Test Summarization
text_file = "president georgi parvanov summoned france 's ambassador on wednesday in a show of displeasure over comments from french president jacques chirac chiding east european nations for their support of washington on the issue of iraq ." # Add text here which you want to summarize
txt = clean_text(text_file)
summarize(txt.lower())