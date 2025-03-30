import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecipeDataProcessor:
    def _init_(self, csv_path):
        """
        Initialize data processor with recipe CSV
        
        :param csv_path: Path to recipes CSV file
        """
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load recipes
        self.recipes_df = pd.read_csv(csv_path)
        
        # Preprocess recipes
        self._preprocess_recipes()
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.ingredient_matrix = self.vectorizer.fit_transform(
            self.recipes_df['ingredients']
        )
    
    def _preprocess_recipes(self):
        """
        Preprocess recipe data
        - Clean text
        - Normalize ingredients
        """
        # Lowercase and strip whitespace
        text_columns = ['name', 'ingredients', 'instructions']
        for col in text_columns:
            self.recipes_df[col] = self.recipes_df[col].str.lower().str.strip()
        
        # Remove duplicates
        self.recipes_df.drop_duplicates(subset=['name'], inplace=True)
    
    def find_similar_recipes(self, query, top_n=3):
        """
        Find similar recipes based on ingredient similarity
        
        :param query: Search query or ingredients
        :param top_n: Number of similar recipes to return
        :return: DataFrame of similar recipes
        """
        # Vectorize query
        query_vectorized = self.vectorizer.transform([query.lower()])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vectorized, self.ingredient_matrix)[0]
        
        # Get top N similar recipes
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        return self.recipes_df.iloc[top_indices]
    
    def extract_ingredients(self, recipe_text):
        """
        Extract ingredients using spaCy
        
        :param recipe_text: Text to extract ingredients from
        :return: List of ingredients
        """
        doc = self.nlp(recipe_text.lower())
        
        # Extract potential ingredients
        ingredients = []
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'FOOD']:
                ingredients.append(ent.text)
        
        return ingredients

# Example usage
if _name_ == '_main_':
    processor = RecipeDataProcessor('data/recipes.csv')
    
    # Find similar recipes
    similar_recipes = processor.find_similar_recipes('chicken')
    print("Similar Recipes:")
    print(similar_recipes)
    
    # Extract ingredients
    ingredients = processor.extract_ingredients('Make a delicious chicken pasta with garlic and herbs')
    print("\nExtracted Ingredients:")
    print(ingredients)