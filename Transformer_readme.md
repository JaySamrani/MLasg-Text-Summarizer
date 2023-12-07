# Transformer text summarizer

## Data Preprocessing: 

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


