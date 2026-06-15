import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQs
try:
    with open("faqs.json", "r", encoding="utf-8") as file:
        faqs = json.load(file)
except FileNotFoundError:
    print("Error: faqs.json file not found!")
    exit()

# Extract questions
questions = [faq["question"] for faq in faqs]

# Create TF-IDF model
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
faq_vectors = vectorizer.fit_transform(questions)

# Function to get chatbot response
def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I could not find a relevant answer."

    return faqs[best_match_index]["answer"]

# Chatbot loop
print("=" * 40)
print("FAQ CHATBOT")
print("Type 'exit' to quit")
print("=" * 40)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Bot:", response)