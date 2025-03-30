import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
import pickle

class RecipeEmbeddings:
    def _init_(self, recipes_csv):
        """
        Generate and manage recipe embeddings
        
        :param recipes_csv: Path to recipes CSV
        """
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load recipes
        self.recipes_df = pd.read_csv(recipes_csv)
        
        # Tokenize recipes
        self.recipe_tokens = self._tokenize_recipes()
        
        # Train Word2Vec model
        self.model = self._train_word2vec()
    
    def _tokenize_recipes(self):
        """
        Tokenize recipe ingredients and instructions
        
        :return: List of tokenized recipes
        """
        recipe_tokens = []
        
        for _, row in self.recipes_df.iterrows():
            # Combine ingredients and instructions
            full_text = f"{row['ingredients']} {row['instructions']}"
            
            # Tokenize using spaCy
            doc = self.nlp(full_text.lower())
            tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
            
            recipe_tokens.append(tokens)
        
        return recipe_tokens
    
    def _train_word2vec(self, vector_size=100, window=5, min_count=1):
        """
        Train Word2Vec model on recipe tokens
        
        :param vector_size: Dimensionality of the word vectors
        :param window: Maximum distance between current and predicted word
        :param min_count: Minimum word count threshold
        :return: Trained Word2Vec model
        """
        model = Word2Vec(
            sentences=self.recipe_tokens, 
            vector_size=vector_size, 
            window=window, 
            min_count=min_count, 
            workers=4
        )
        
        return model
    
    def get_recipe_vector(self, ingredients):
        """
        Generate vector representation of ingredients
        
        :param ingredients: List of ingredients or ingredient string
        :return: Average vector of ingredient words
        """
        # Tokenize ingredients
        if isinstance(ingredients, str):
            doc = self.nlp(ingredients.lower())
            tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        else:
            tokens = ingredients
        
        # Get word vectors
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.model.wv[token])
            except KeyError:
                # Skip tokens not in vocabulary
                continue
        
        # Return average vector or zero vector if no vectors found
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)
    
    def find_similar_ingredients(self, ingredients, top_n=5):
        """
        Find similar ingredients based on word embeddings
        
        :param ingredients: List of ingredients
        :param top_n: Number of similar ingredients to return
        :return: List of similar ingredients
        """
        # Get ingredient vector
        ingredient_vector = self.get_recipe_vector(ingredients)
        
        # Find most similar words
        similar_words = self.model.wv.similar_by_vector(ingredient_vector, topn=top_n)
        
        return [word for word, _ in similar_words]
    
    def save_embeddings(self, filepath='models/recipe_embeddings.pkl'):
        """
        Save trained embeddings to file
        
        :param filepath: Path to save embeddings
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'recipes': self.recipes_df
            }, f)
    
    def load_embeddings(self, filepath='models/recipe_embeddings.pkl'):
        """
        Load pre-trained embeddings from file
        
        :param filepath: Path to load embeddings from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.recipes_df = data['recipes']

# Example usage
if _name_ == '_main_':
    embeddings = RecipeEmbeddings('data/recipes.csv')
    
    # Get ingredient vector
    vector = embeddings.get_recipe_vector(['chicken', 'pasta'])
    print("Ingredient Vector:", vector)
    
    # Find similar ingredients
    similar_ingredients = embeddings.find_similar_ingredients(['chicken'])
    print("\nSimilar Ingredients:")
    print(similar_ingredients)
    
    # Save embeddings
    embeddings.save_embeddings()