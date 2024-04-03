from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
import string
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from difflib import SequenceMatcher

app = Flask(__name__)

# Define the path for storing and loading the user input knowledgebase
JSON_FILE_PATH = 'knowledgebase.json'

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('popular')
nltk.download('maxent_ne_chunker')
nltk.download('movie_reviews')
nltk.download('names')
nltk.download('reuters')
nltk.download('sentiwordnet')
nltk.download('shakespeare')
nltk.download('sinica_treebank')
nltk.download('state_union')
nltk.download('treebank')
nltk.download('udhr')

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to load existing user input knowledgebase
def load_responses(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading responses: {e}")
        return {}

# Function to preprocess input text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text.lower())
    filtered_sentences = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words and word not in string.punctuation]) for sentence in sentences]
    return filtered_sentences

# Function to find the most similar word
def get_most_similar_word(input_word, word_list):
    input_lemma = lemmatizer.lemmatize(input_word.lower())
    similarity_scores = {word: nltk.edit_distance(input_lemma, word.lower()) for word in word_list}
    most_similar_word = min(similarity_scores, key=similarity_scores.get)
    return most_similar_word

# Function to generate optimal response
def generate_optimal_response(responses, user_input):
    try:
        most_similar_word = get_most_similar_word(user_input, responses.keys())

        if most_similar_word and most_similar_word in responses:
            response_data = responses[most_similar_word]
            response_type = response_data.get("response_type")
            bot_responses = response_data.get("bot_response")

            if bot_responses and response_type:
                if response_type in ["greeting", "tv interview", "confirmation", "weather forecast", "assist me", 
                                    "inquiry", "opinion", "experience", "self-introduction", "favourite tv show", 
                                    "favourite color", "favourite cuisine", "favourite hobby", "favourite holiday", 
                                    "favourite movie", "favourite music genre", "favourite season", "favourite sport", 
                                    "favourite Tv show", "free time activities", "favourite book", "favourite movies", 
                                    "book recommendation", "weekend experience", "restaurant recommendation", 
                                    "identify", "tv interview feedback", "weather outside", "religious", "my religion"]:
                    # Randomly choose a response and add punctuation if missing
                    combined_response = random.choice(bot_responses)
                    if combined_response and combined_response[-1] not in ['!', '.', '?']:
                        combined_response += random.choice(['!', '.', '?'])
                elif response_type == "python_def" or response_type == "python_code":
                    # Combine all responses into one preserving order
                    combined_response = ' '.join(bot_responses)
                else:
                    # Shuffle bot_responses before selecting
                    random.shuffle(bot_responses)
                    
                    # Combine bot responses until reaching 100 words
                    combined_response = ''
                    total_words = 0
                    for response in bot_responses:
                        combined_response += response + ' '
                        total_words += len(response.split())
                        if total_words >= 70:
                            break

                    return combined_response.strip()

                return combined_response.strip()
    except Exception as e:
        print(f"Error generating optimal response: {e}")
    return None

# Function to handle user input with multiple questions
def handle_user_input(prev_combined_response=None):
    try:
        user_input = request.json['user_input'].lower()

        if user_input == 'exit':
            print("Chat bot exiting.")
            return None
        elif prev_combined_response and re.search(r'^\d+\.\s+', user_input):
            # Convert the input points to the correct format
            split_input = re.split(r'\d+\.\s+', user_input)
            cleaned_input = [item.strip() for item in split_input if item.strip()]
            if cleaned_input:
                cleaned_input[0] = prev_combined_response
                return cleaned_input
            else:
                return None
        else:
            return re.split(r'\band\b', user_input)  # Splitting input using regex pattern '&'
    except Exception as e:
        print(f"Error handling user input: {e}")
        return []

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling user input and generating responses
@app.route('/ask', methods=['POST'])
def ask():
    responses = load_responses(JSON_FILE_PATH)
    if not responses:
        return jsonify({'response': 'Error: No responses loaded.'})

    user_inputs = handle_user_input()
    if not user_inputs:
        return jsonify({'response': 'Chat bot exiting.'})

    combined_responses = []  # List to store all combined responses

    for user_input in user_inputs:
        # Generate optimal response for each user input
        combined_response = generate_optimal_response(responses, user_input)
        if combined_response:
            combined_responses.append(combined_response)  # Append combined response to the list
        else:
            print("Bot: I don't know the answer to one of the questions.")

    return jsonify({'response': ' '.join(combined_responses)})

if __name__ == "__main__":
    app.run(debug=True)