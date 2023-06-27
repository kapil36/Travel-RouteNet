# Travel-RouteNet
The code aims to extract valuable features from raw text data to identify the source and destination cities mentioned in messages. It utilizes pre-trained word embeddings, LSTM layers for sequence processing, and a supervised learning approach to achieve this task.

Certainly! Here's a detailed explanation of the code in five paragraphs, along with subheadings, an introduction, and a conclusion.

Introduction:
The provided code implements a feature extraction model that focuses on identifying the source city and destination city from raw text data. It utilizes a combination of natural language processing techniques and deep learning algorithms to preprocess the data, train a model, and make predictions. The goal is to extract valuable information from text messages and accurately determine the source and destination cities mentioned within.

Data Preprocessing:
The code begins by importing the necessary libraries and reading a CSV file containing the text messages, source cities, and destination cities. It preprocesses the data in several steps. First, it converts the text and label columns to lowercase and removes leading and trailing spaces. This standardizes the data and avoids any inconsistencies due to capitalization or spacing. Next, the labels are encoded using scikit-learn's LabelEncoder, which assigns a unique numerical value to each distinct label. Then, the texts are tokenized using the Tokenizer from TensorFlow, which breaks down each text into a sequence of individual words or tokens. Finally, the sequences are padded to a consistent length using TensorFlow's pad_sequences, ensuring that all input sequences have the same dimensions.

Word Embeddings:
To enhance the model's understanding of the textual data, pre-trained GloVe word embeddings are utilized. The code loads these embeddings using the gensim library. Word embeddings are dense vector representations that capture semantic relationships between words. An embedding matrix is created based on the loaded word embeddings, which maps each word in the vocabulary to its corresponding embedding vector. This matrix serves as the initial weights for the embedding layer in the model.

Model Architecture:
The core of the code revolves around building the LSTM-based model for feature extraction. The model is created using the Sequential class from TensorFlow's Keras API. The architecture consists of multiple layers:

The first layer is an embedding layer that uses the embedding matrix. This layer maps the input word indices to their respective embedding vectors.
Bidirectional LSTM layers are then added to capture the context from both directions of the input sequences. The bidirectional nature helps the model to understand the dependencies and patterns in the text more effectively.
Dropout layers are inserted after the LSTM layers to prevent overfitting by randomly dropping a fraction of the connections during training.
Dense layers with appropriate activation functions are included for classification purposes. The output layer uses the softmax activation function to produce the predicted probabilities for each possible label.
Model Training and Prediction:
After constructing the model, the code splits the preprocessed data into training and testing sets using scikit-learn's train_test_split function. The model is then compiled with a suitable loss function, optimizer, and evaluation metric. It is trained on the training data using the fit method, specifying the number of epochs and batch size. Once the training is complete, the model is used to predict labels for predefined strings. These strings are tokenized, padded, and passed through the model to obtain the predicted label. The predicted label is then converted back to its original form using the label encoder, and both the predicted label and the original message are printed.

Model Summary and Conclusion:
To conclude, the code provides a summary of the model's architecture, including the number of parameters and layer configurations, using the summary method. It offers an overview of the model's structure and complexity. The implementation demonstrates a practical approach to extract relevant features from text data using deep learning techniques, specifically LSTM networks. By training the model on labeled data and leveraging pre-trained word embeddings, it achieves accurate predictions of the source and destination cities mentioned in text messages, facilitating effective feature extraction for further analysis or applications in the domain of city routing or travel planning.

Overall, the code combines data preprocessing, word embeddings, LSTM-based model architecture, and training/prediction stages to extract valuable information from text data and identify source and destination cities accurately. This feature extraction capability can be useful in various applications such as travel planning, transportation services, or city-specific analysis.

<img width="693" alt="image" src="https://github.com/kapil36/Travel-RouteNet/assets/64062901/4c3821fd-ccfb-4c86-ae2d-12756455d5b0">

You can Use this model according to you're requirement

Contact me - 
36kapil63@gmail.com
