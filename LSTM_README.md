# MLasg-Text-Summarizer
## 1. Data Pre-processing
Overview:
In order to prepare raw data for use in machine learning and natural language processing applications, data preprocessing is an essential step. We describe in this paper the several pretreatment processes we used to prepare the dataset for text summarizing. Our task is to clean and convert the dataset, which is made up of text and summary pairs, into a format that can be used to train a text summarizing model.
Cleaning: 
-	Removal of escape characters (‘\t’, ‘ \r’, ‘ \n’).
-	Removal of multiple consecutive spaces, colons, hyphens, full stop at the end of words.
-	Removal of special characters such as <, >, (), |, &, ©, ø, ', ", ,, ;, ?, ~, and *.
Tokenization:
	Tokenization is the process of splitting the text into individual words or tokens. Tokenization was performed using the spaCy library.
Text Length Filtering: 
-	Selecting text and summary pairs based on maximum text and summary lengths (max_text_len and max_label_len).
-	Shortening the text and summary data to contain only pairs that meet the length criteria.
- 	Post padding is used for short sentences. (length=100 for input paragraph) (length=15 for summary)
Special Tokens: 
The special tokens "START" and "END" were added to the summary data to indicate the start and end of a summary. 
Data Organization:
The cleaned and tokenized text and summary data were organized into a Pandas DataFrame for further analysis and modeling.

## 2. Model Construction:
For model construction, we have used LSTM library from keras.layers in order to get a quick start in model construction. Keras API provides us pre-implemented layers in tensorflow framework.
#### LSTM layer: ![image](https://github.com/JaySamrani/MLasg-Text-Summarizer/assets/111739529/971a9594-988d-4e11-b774-885dd685afe3)
#### Output Dense layer : ![image](https://github.com/JaySamrani/MLasg-Text-Summarizer/assets/111739529/03eba3f3-4de8-4dcb-adca-3d99dac7db27)

In Keras, TimeDistributed is a wrapper used to apply a layer to every temporal slice of an input. When dealing with sequential data, such as sequences or time series, this layer allows you to apply the specified layer to each time step independently. For instance, if you're working with sequences and want to apply a Dense layer to each time step independently, you can use TimeDistributed to achieve this. It's commonly used in recurrent neural networks (RNNs), especially with Long Short-Term Memory (LSTM) networks, to apply a certain layer (like Dense) to every time step's output.

#### Architecture details:
![image](https://github.com/JaySamrani/MLasg-Text-Summarizer/assets/111739529/9e5a357c-4d9a-450c-be48-71f3d5a11667)

## 3. Process:
- Training data consists of – 98000 examples after cleaning
- Validation data consists of – 4500 examples after cleaning
- After Encoding the text using one hot encoding the sequence is sent to LSTM model.

The model consists of 4 LSTM Layers and 2 embedding layers and final dense layer of ‘softmax’ to obtain index from the model. These indices are sent to a decoder to obtain the target words which gives us the summary of the input text. \
Decoder Setup: The decoder is set up to generate the target sequence one word at a time. It takes three inputs: previous hidden state (decoder_state_input_h), previous cell state (decoder_state_input_c), and the hidden states from the encoder (decoder_hidden_state_input). The decoder embedding layer (dec_emb_layer) is used to get the embeddings of the decoder sequence (decoder_inputs). The LSTM layer (decoder_lstm) takes these embeddings and the initial states to produce the decoder outputs (decoder_outputs2) and the new internal states of the LSTM (state_h2 and state_c2).\
## 4. Post Processing: 
The decoder model is defined using the inputs and outputs from the decoder setup. It takes the decoder inputs and the hidden states from the encoder and produces the decoder outputs and new internal states. The dense softmax layer (decoder_dense) is applied to the decoder outputs to generate a probability distribution over the target vocabulary.

## 5. Contribution:
###### Jay Samrani : Model construction, model training, helping Narendra in decoding sequence. 
###### Manjeet : Loss and accuracy management, Data Pre-processing: Data cleaning, Tokenization, text length filtering, adding special tokens. 
###### Narendra : Post-processing, Output sequence decoding 
