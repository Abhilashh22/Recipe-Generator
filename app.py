from flask import Flask, render_template, request, jsonify
from src.recipe_generation import RecipeGenerator
from src.intent_processing import RecipeIntentClassifier
from src.data_processing import RecipeDataProcessor

app = Flask(_name_)

# Initialize components
recipe_generator = RecipeGenerator('data/recipes.csv')
intent_classifier = RecipeIntentClassifier()
data_processor = RecipeDataProcessor('data/recipes.csv')

@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    """
    Generate a recipe based on user input
    """
    # Get user input
    user_input = request.form.get('query', '')
    
    try:
        # Classify intent
        intent_result = intent_classifier.classify_intent(user_input)
        entities = intent_result['entities']
        
        # Generate recipe
        recipe = recipe_generator.generate_recipe(
            base_ingredients=entities['ingredients'],
            cuisine=entities['cuisine'],
            difficulty=entities['difficulty']
        )
        
        # Find similar recipes
        similar_recipes = data_processor.find_similar_recipes(user_input)
        
        # Generate variations
        variations = recipe_generator.generate_recipe_variations(recipe)
        
        return jsonify({
            'recipe': recipe,
            'similar_recipes': similar_recipes.to_dict(orient='records'),
            'variations': variations,
            'response': intent_classifier.generate_response(intent_result)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'response': 'Sorry, I could not generate a recipe. Please try again.'
        }), 500

@app.route('/recipe_variations', methods=['POST'])
def get_recipe_variations():
    """
    Generate recipe variations
    """
    recipe_data = request.json
    variations = recipe_generator.generate_recipe_variations(recipe_data)
    return jsonify({'variations': variations})

@app.route('/find_similar_recipes', methods=['POST'])
def find_similar_recipes():
    """
    Find similar recipes based on ingredients
    """
    ingredients = request.form.get('ingredients', '')
    similar_recipes = data_processor.find_similar_recipes(ingredients)
    return jsonify({
        'similar_recipes': similar_recipes.to_dict(orient='records')
    })

if _name_ == '_main_':
    app.run(debug=True, port=5000)