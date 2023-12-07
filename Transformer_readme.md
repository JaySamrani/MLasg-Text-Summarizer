# Transformer text summarizer

## 1. Data Preprocessing: 

#### 1. Data Loading and Cleaning:
The script loads the dataset from an Excel file and removes unnecessary columns ('Source', 'Time', 'Publish Date') from the dataset, keeping only the 'Short' and 'Headline' columns for article text and corresponding summaries, respectively.

#### 2. Text Preprocessing: 
It includes a function, preprocess(), employing regular expressions to clean the text. Specifically, it removes patterns like '&.[1-9]+;' which might interfere with subsequent processing steps.

#### 3. Tokenization and Padding: 
The script utilizes the Tokenizer class from Keras to tokenize both the article text and summaries separately. It converts the text data into sequences of integers and then pads these sequences to fixed lengths using Keras' pad_sequences function, ensuring uniformity in sequence length.

#### 4. Vocabulary Size Calculation:
The vocabulary sizes for both the encoder and decoder are computed based on the tokenized sequences. This is crucial for setting up the model architecture later.

#### 5. Dataset Preparation:
The processed data is converted into TensorFlow Dataset objects, allowing for convenient batching, shuffling, and handling during the model training phase.


## 2. Model conponents construction:
### Positional Encoding:
get_angles and positional_encoding functions create positional encodings used to inject positional information into the input sequences.
### - Masking:
create_padding_mask and create_look_ahead_mask functions generate masks to handle padding and look-ahead sequences, respectively.
### -  Scaled Dot-Product Attention:
scaled_dot_product_attention computes attention weights and output based on scaled dot-product attention mechanism.
### -  Multi-Head Attention:
MultiHeadAttention class implements multi-head attention with parallel attention heads.
### -  Feed-Forward Network:

point_wise_feed_forward_network function defines a feed-forward neural network with two dense layers.
### -  Encoder and Decoder Layers:

EncoderLayer and DecoderLayer classes consist of multi-head attention, feed-forward networks, layer normalization, and dropout for encoder and decoder components.
### -  Encoder and Decoder:

Encoder and Decoder classes stack multiple encoder and decoder layers respectively, along with embeddings and positional encodings.
### -  Transformer Model:

Transformer class integrates the encoder and decoder, including a final dense layer for output.
