import random
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RecipeGenerator:
    def _init_(self, recipes_csv, model_name='gpt2'):
        """
        Initialize Recipe Generator
        
        :param recipes_csv: Path to CSV with recipes
        :param model_name: Pretrained model name
        """
        # Load recipes
        self.recipes_df = pd.read_csv(recipes_csv)
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_recipe(self, 
                         base_ingredients=None, 
                         cuisine=None, 
                         difficulty='medium'):
        """
        Generate a new recipe
        
        :param base_ingredients: List of base ingredients
        :param cuisine: Specific cuisine type
        :param difficulty: Recipe difficulty level
        :return: Generated recipe dictionary
        """
        # Filter recipes based on parameters
        filtered_recipes = self.recipes_df.copy()
        
        if base_ingredients:
            filtered_recipes = filtered_recipes[
                filtered_recipes['ingredients'].apply(
                    lambda x: all(ing.lower() in x.lower() for ing in base_ingredients)
                )
            ]
        
        if cuisine:
            filtered_recipes = filtered_recipes[
                filtered_recipes['cuisine'].str.lower() == cuisine.lower()
            ]
        
        # Fallback to full dataset if no matching recipes
        if filtered_recipes.empty:
            filtered_recipes = self.recipes_df
        
        # Select a random recipe for inspiration
        inspiration = filtered_recipes.sample(1).iloc[0]
        
        # Prepare prompt for recipe generation
        prompt = f"Create a {difficulty} difficulty {cuisine or ''} recipe using {inspiration['ingredients']}: "
        
        # Generate recipe instructions
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs, 
            max_length=300, 
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        
        # Decode generated text
        generated_instructions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Construct new recipe
        new_recipe = {
            'name': f"AI Generated {inspiration['name']}",
            'ingredients': inspiration['ingredients'],
            'instructions': generated_instructions,
            'cuisine': cuisine or inspiration['cuisine'],
            'difficulty': difficulty
        }
        
        return new_recipe
    
    def generate_recipe_variations(self, recipe, num_variations=3):
        """
        Generate recipe variations
        
        :param recipe: Original recipe dictionary
        :param num_variations: Number of variations to generate
        :return: List of recipe variations
        """
        variations = []
        
        # Variation strategies
        strategies = [
            self._swap_ingredients,
            self._modify_cooking_method,
            self._add_spices
        ]
        
        for _ in range(num_variations):
            # Randomly select variation strategy
            strategy = random.choice(strategies)
            variation = strategy(recipe)
            variations.append(variation)
        
        return variations
    
    def _swap_ingredients(self, recipe):
        """
        Swap ingredients with alternatives
        """
        ingredient_map = {
            'chicken': ['tofu', 'turkey', 'fish'],
            'beef': ['lamb', 'pork', 'mushrooms'],
            'pasta': ['rice', 'quinoa', 'zucchini noodles']
        }
        
        # Create a copy of the recipe
        variation = recipe.copy()
        
        # Modify ingredients
        ingredients = variation['ingredients'].split(',')
        for i, ingredient in enumerate(ingredients):
            for base, alternatives in ingredient_map.items():
                if base in ingredient.lower():
                    ingredients[i] = random.choice(alternatives)
        
        variation['ingredients'] = ', '.join(ingredients)
        variation['name'] = f"Variation of {variation['name']}"
        
        return variation
    
    def _modify_cooking_method(self, recipe):
        """
        Modify cooking method
        """
        cooking_methods = ['baked', 'grilled', 'stir-fried', 'steamed', 'roasted']
        
        variation = recipe.copy()
        variation['instructions'] = (
            f"Instead of the original method, {random.choice(cooking_methods)} "
            f"the ingredients according to the original recipe."
        )
        variation['name'] = f"Alternative Method {variation['name']}"
        
        return variation
    
    def _add_spices(self, recipe):
        """
        Add new spices to the recipe
        """
        spices = ['cumin', 'paprika', 'turmeric', 'oregano', 'thyme', 'rosemary']
        
        variation = recipe.copy()
        new_spice = random.choice(spices)
        variation['ingredients'] += f", {new_spice}"
        variation['name'] = f"Spiced {variation['name']}"
        
        return variation

# Example usage
if __name__ == '_main_':
    generator = RecipeGenerator('data/recipes.csv')
    
    # Generate a recipe
    new_recipe = generator.generate_recipe(        base_ingredients=['chicken', 'pasta'], 
        cuisine='Italian',
        difficulty='medium'
    )
    
    print("Generated Recipe:")
    for key, value in new_recipe.items():
        print(f"{key.capitalize()}: {value}")
    
    # Generate variations
    variations = generator.generate_recipe_variations(new_recipe)
    print("\nRecipe Variations:")
    for i, variation in enumerate(variations, 1):
        print(f"\nVariation {i}:")
        for key, value in variation.items():
            print(f"{key.capitalize()}: {value}")