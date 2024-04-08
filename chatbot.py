import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow

# Initializing WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from json
intents = json.loads(open("intents.json").read())

# Load preprocessed words and classes from pickle files
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load pre-trained model from file
model = tensorflow.keras.models.load_model("chatbot_model.keras")

# Function to tokenize and lemmatize input sentences
def clean_up_sentence(sentence):
    # Tokenize the input sentence into words
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word in the tokenized sentence
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    # Return the lemmatized words
    return sentence_words

# Function to convert sentence into bag of words representation
def bag_of_words(sentence):
    # Tokenize and lemmatize the input sentence
    sentence_words = clean_up_sentence(sentence)
    # Initialize a bag of words with zeros, with length equal to the number of words
    bag = [0] * len(words)
    # Iterage through each word in tokenized sentence
    for w in sentence_words:
        # Enumerate each word in words
        for i, word in enumerate(words):
            # Check if the word is equal to the word in sentence_words
            if word == w:
                # If the word exists, set the corresponding index in the bag to 1
                bag[i] = 1
    # Return the bag of words as a numpy array
    return np.array(bag)

# Function to predict the intent of a given sentence
def predict_class(sentence):
    # Convert the sentence into a bag of words
    bag = bag_of_words(sentence)
    # Predict the probability distribution of intents using the pre-trained model
    res = model.predict(np.array([bag]))[0]
    # Define a error threshold to filter low-probability predictions
    ERROR_THRESHOLD = 0.25
    # Filter out predictions with probability below the threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort the filtered results by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    # Extract predicted intents along with their probabilities
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # Return the list of predicted intents
    return return_list

# Function to get a random response based on predicted intent
def get_response(intents_list, intents_json):
    # Extract the predicted intent from the intents list
    tag = intents_list[0]['intent']
    # Retrieve the list of intents from the intents JSON data
    list_of_intents = intents_json['intents']
    # Find the corresponding intent in the intents JSON data
    for i in list_of_intents:
        if i['tag'] == tag:
            # Select a random response from the response associated with the predicted intent
            result = random.choice(i['responses'])
            break
    # Return the selected response
    return result

print("Bot is running")

# Continuous loop to receive and respond to user input
while True:
    # Get user input
    message = input("")
    # Predict intent of the input
    ints = predict_class(message)
    # Get a response based on predicted intent
    res = get_response(ints, intents)
    # Print the response
    print(res)
