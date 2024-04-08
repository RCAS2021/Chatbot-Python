import random
import json
# Serialization
import pickle
import numpy as np
import tensorflow as tf
import nltk
# Reduce word for its stem
from nltk.stem import WordNetLemmatizer

# Download NTLK resources (punkt), the English tokenizer and (WordNet), used for lemmatization in NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Initializing the lemmatizer
lemmatizer = WordNetLemmatizer()

# Reading json file content as text, returning a dictionary
intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Creating document with each words specifying it's tag, example:
# [(['hey'], 'greetings'), (['hello'], 'greetings'),  (['see', 'you', 'later'], 'goodbye'), ... ]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Splits in different words
        word_list = nltk.word_tokenize(pattern)
        # Add collection of words to words list -> extend puts each word, using append puts the entire list
        words.extend(word_list)
        # This wordList belongs to this tag
        # Check if intent tag is in classes list
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize each word in words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Sort words and remove duplicates
words = sorted(set(words))

classes = sorted(set(classes))

# Save into files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
# Create a template with as much 0 as there are classes
output_empty = [0] * len(classes)

# Iterating through each document
for document in documents:
    # Empty list for pag of words for the current document
    bag = []
    # Extract the word patterns from the current document
    word_patterns = document[0]
    # Lemmatize and convert all words to lowercase
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Iterating through each word
    for word in words:
        # Check if the word exists in the word patterns
        # If it does, append 1 to the bag otherwise append 0
        bag.append(1) if word in word_patterns else bag.append(0)

    # Copying list
    output_row = list(output_empty)
    # Find the index of the document's class/tag in the list of classes
    # Update the corresponding index in the output row to 1 to indicate the class/tag of the document
    output_row[classes.index(document[1])] = 1
    # Append the bag of words and the output row to the training data
    training.append(bag + output_row)

# Shuffle data
random.shuffle(training)
# Transform in numpy array
training = np.array(training)

# Train X receives training data up to the length of the words list (bag of rows)
train_x = training[:, :len(words)]
# Train y receives training data starting from length of the words list (output rows)
train_y = training[:, len(words):]

# Building model
model = tf.keras.Sequential() # Define a sequential model, which allows for linear stack of layers
# Add input layer with 128 neurons, input shape is determined by the length of the first training sample
# Activation function used is ReLU (Rectified Linear Unit)
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add a dropout layer with a rate of 0.5 to prevent overfitting
model.add(tf.keras.layers.Dropout(0.5))
# Add a hidden layer with 64 neurons and ReLU activation function
model.add(tf.keras.layers.Dense(64, activation='relu'))
# Add another dropout layer to prevent overfitting
model.add(tf.keras.layers.Dropout(0.5))
#Add the output layer with neurons equal to the number of classes, using softmax activation for multi-class classification
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Define Stochastic Gradient Descent (SGD) optimizer with specified learning rate, momentum and Nesterov momentum
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# Compile the model, specifying loss function, optimizer and evaluation metric
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model using training data
# epochs: Number of training iterations over the entire dataset
# batch_size: Number of samples per gradient update
# verbose: Verbosity mode(0 = silent, 1 = progress bar, 2 = one line per epoch)
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Saving the trained model to a file containing .keras format
# Saving the training history for visualization purposes
model.save('chatbot_model.keras', hist)

print("Done")
