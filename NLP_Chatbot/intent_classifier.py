"""
Intent Classifier using NLP_Recognition Preprocessing
Handles tokenization, normalization, and intent matching
"""

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class IntentClassifier:
    """
    Classifies user input to the best matching intent
    Uses TF-IDF vectorization and cosine similarity
    """

    def __init__(self, knowledge_base):
        """Initialize the classifier with knowledge base"""
        self.knowledge_base = knowledge_base
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self._build_intent_vectors()

    def _preprocess_text(self, text):
        """Preprocess text: lowercase, remove special chars, tokenize, remove stopwords, stem"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Split into words
        words = text.split()

        # Remove stopwords and stem
        words = [
            self.stemmer.stem(word)
            for word in words
            if word not in self.stop_words and len(word) > 0
        ]

        return ' '.join(words)

    def _build_intent_vectors(self):
        """Build TF-IDF vectors for all intent patterns"""
        self.intent_patterns = {}
        all_patterns = []

        # Collect all patterns from knowledge base
        for intent, data in self.knowledge_base.items():
            patterns = data['patterns']
            preprocessed_patterns = [
                self._preprocess_text(pattern)
                for pattern in patterns
            ]
            self.intent_patterns[intent] = preprocessed_patterns
            all_patterns.extend(preprocessed_patterns)

        # Build TF-IDF vectors
        self.intent_vectors = self.vectorizer.fit_transform(all_patterns)

    def classify_intent(self, user_input, threshold=0.3):
        """Classify user input to best matching intent"""
        # Preprocess user input
        processed_input = self._preprocess_text(user_input)

        if not processed_input:
            return None, 0.0

        # Vectorize user input
        user_vector = self.vectorizer.transform([processed_input])

        # Calculate similarity with all patterns
        similarities = cosine_similarity(user_vector, self.intent_vectors)[0]

        # Find the best match
        best_score = np.max(similarities)

        if best_score < threshold:
            return None, best_score

        # Find which intent has the best match
        best_pattern_idx = np.argmax(similarities)

        # Map pattern index back to intent
        current_idx = 0
        for intent, patterns in self.intent_patterns.items():
            if best_pattern_idx < current_idx + len(patterns):
                return intent, best_score
            current_idx += len(patterns)

        return None, best_score


def get_classifier(knowledge_base):
    """Factory function to create and return an intent classifier"""
    return IntentClassifier(knowledge_base)