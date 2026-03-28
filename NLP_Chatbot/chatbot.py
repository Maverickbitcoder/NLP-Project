"""
Main Chatbot Class
Handles conversation flow and response generation
"""

import random
from intent_classifier import get_classifier
from knowledge_base import KNOWLEDGE_BASE, FALLBACK_RESPONSES


class NLPChatbot:
    """
    Intent-based chatbot for NLP_Recognition and AI queries
    Uses NLP_Recognition preprocessing and similarity-based intent matching
    """

    def __init__(self, knowledge_base=None):
        """Initialize the chatbot"""
        self.knowledge_base = knowledge_base or KNOWLEDGE_BASE
        self.classifier = get_classifier(self.knowledge_base)
        self.conversation_history = []

    def get_response(self, user_input):
        """Generate a response to user input"""
        # Classify the intent
        intent, confidence = self.classifier.classify_intent(user_input)

        # Store in conversation history
        self.conversation_history.append({
            'user': user_input,
            'intent': intent,
            'confidence': confidence
        })

        # Get response based on intent
        if intent and intent in self.knowledge_base:
            response = random.choice(self.knowledge_base[intent]['responses'])
        else:
            response = random.choice(FALLBACK_RESPONSES)

        return response, intent, confidence

    def get_conversation_history(self):
        """Return conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []