# Transformer text summarizer

## 1. Data Preprocessing: 

#### -  Data Loading and Cleaning:
The script loads the dataset from an Excel file and removes unnecessary columns ('Source', 'Time', 'Publish Date') from the dataset, keeping only the 'Short' and 'Headline' columns for article text and corresponding summaries, respectively.

#### -  Text Preprocessing: 
It includes a function, preprocess(), employing regular expressions to clean the text. Specifically, it removes patterns like '&.[1-9]+;' which might interfere with subsequent processing steps.

#### -  Tokenization and Padding: 
The script utilizes the Tokenizer class from Keras to tokenize both the article text and summaries separately. It converts the text data into sequences of integers and then pads these sequences to fixed lengths using Keras' pad_sequences function, ensuring uniformity in sequence length.

#### -  Vocabulary Size Calculation:
The vocabulary sizes for both the encoder and decoder are computed based on the tokenized sequences. This is crucial for setting up the model architecture later.

#### -  Dataset Preparation:
The processed data is converted into TensorFlow Dataset objects, allowing for convenient batching, shuffling, and handling during the model training phase.


## 2. Model conponents construction:
### Positional Encoding:
get_angles and positional_encoding functions create positional encodings used to inject positional information into the input sequences.
![image](https://github.com/JaySamrani/MLasg-Text-Summarizer/assets/111739529/e3e9b35b-9f66-4539-9973-c7373fe4f524)

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
![image](https://github.com/JaySamrani/MLasg-Text-Summarizer/assets/111739529/aa6f7d66-bd90-48f0-a15e-f7b7a93789b4)

## 3. Process:
The Transformer-based text summarization process involves several key steps, starting with the preparation and preprocessing of the dataset. The code provided loads and cleanses the Inshorts Cleaned News dataset, preparing the 'Short' and 'Headline' columns for article text and corresponding summaries. It includes text preprocessing steps such as tokenization, padding sequences to fixed lengths, and creating vocabularies for both the encoder and decoder. The implementation focuses on crucial elements like positional encoding, masking mechanisms for handling padding and future tokens, and attention mechanisms (such as scaled dot-product and multi-head attention). These are integrated into encoder and decoder layers, forming the backbone of the Transformer model. Additionally, the code establishes a custom learning rate schedule and optimizer, essential for training the model effectively. Overall, this process lays the foundation for training a Transformer model specifically tailored for text summarization tasks using the Inshorts Cleaned News dataset.

## 4. Post-processing:
After the Transformer model completes the summarization process, post-processing steps are crucial for refining the generated summaries. The post-processing phase involves tasks such as detokenization, which reverses the tokenization process to reconstruct human-readable summaries. This step aims to transform the model-generated tokenized sequences back into natural language sentences or phrases. Additionally, the generated summaries might undergo polishing procedures like grammar checks, coherence evaluations, or length adjustments to ensure they are fluent, accurate, and align with the intended summarization objectives. Post-processing plays a pivotal role in enhancing the quality and readability of the final summaries produced by the Transformer-based text summarization model.

## 5. Contribution:
### Manjeet: Data preprocessing, masking. 
### Jay Samrani: Post processing, Evaluation, testing.
### Narendra Kumar: Architecture construction,  Learning rate schedular
