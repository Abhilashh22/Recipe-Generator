import spacy
import json
import rasa
from rasa.core.agent import Agent
from rasa.shared.utils.io import json_to_string

class RecipeIntentClassifier:
    def _init_(self, 
                 model_path='rasa/models', 
                 config_path='rasa/config.yml'):
        """
        Initialize Rasa Intent Classifier
        
        :param model_path: Path to trained Rasa model
        :param config_path: Path to Rasa configuration
        """
        # Load spaCy for additional NLP processing
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load Rasa model
        self.agent = Agent.load(model_path)
    
    def classify_intent(self, user_message):
        """
        Classify user intent for recipe generation
        
        :param user_message: User's text input
        :return: Classified intent and extracted entities
        """
        # Process message with Rasa
        result = self.agent.parse_message(user_message)
        
        # Extract intent and confidence
        intent = result.get('intent', {}).get('name', 'default')
        confidence = result.get('intent', {}).get('confidence', 0.0)
        
        # Process with spaCy for additional entity extraction
        doc = self.nlp(user_message)
        
        # Extract entities
        entities = {
            'ingredients': [],
            'cuisine': None,
            'difficulty': None
        }
        
        # Extract ingredients and cuisine
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'FOOD']:
                entities['ingredients'].append(ent.text)
            elif ent.label_ in ['GPE', 'ORG']:
                entities['cuisine'] = ent.text
        
        # Determine difficulty
        difficulty_map = {
            'easy': ['easy', 'simple', 'beginner'],
            'hard': ['complex', 'difficult', 'advanced']
        }
        
        for diff, keywords in difficulty_map.items():
            if any(keyword in user_message.lower() for keyword in keywords):
                entities['difficulty'] = diff
                break
        
        # Default to medium difficulty
        if not entities['difficulty']:
            entities['difficulty'] = 'medium'
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities
        }
    
    def generate_response(self, intent_result):
        """
        Generate appropriate response based on intent
        
        :param intent_result: Result from intent classification
        :return: Response text
        """
        intent = intent_result['intent']
        entities = intent_result['entities']
        
        responses = {
            'generate_recipe': (
                f"I'll generate a {entities['difficulty']} recipe "
                f"with {', '.join(entities['ingredients']) or 'ingredients'}"
            ),
            'recipe_variation': "I'll create some recipe variations for you.",
            'find_similar_recipes': "I'll find similar recipes based on your input.",
            'default': "I'm not sure what recipe you're looking for. Can you be more specific?"
        }
        
        return responses.get(intent, responses['default'])

# Example usage
if __name__ == '_main_':
    classifier = RecipeIntentClassifier()
    
    # Test intent classification
    test_messages = [
        "I want to make an easy chicken pasta",
        "Can you generate an Italian recipe?",
        "Find me recipes similar to beef stew",
        "Help me create a complex vegetarian dish"
    ]
    
    for message in test_messages:
        print(f"\nMessage: {message}")
        intent_result = classifier.classify_intent(message)
        print("Intent Classification:")
        print(json.dumps(intent_result, indent=2))
        
        response = classifier.generate_response(intent_result)
        print(f"Response: {response}")