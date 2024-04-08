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
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)

print("Done")
