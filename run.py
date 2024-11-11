from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
from openai import OpenAI
from lib import *
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

template_mapping = {
        1: 'Batman-Slapping-Robin',
        2: 'Bike-Fall',
        3: 'Buff-Doge-vs-Cheems',
        4: 'Clown-Applying-Makeup',
        5: 'Distracted-Boyfriend',
        6: 'Drake',
        7: 'Hide-the-Pain-Harold',
        8: 'Monkey-Puppet',
        9: 'One-Does-Not-Simply',
        10: 'Running-Away-Balloon',
        11: 'Sleeping-Shaq',
        12: 'Whisper-and-Goosebumps'
}

meme_location_mapping = {
        'Batman-Slapping-Robin': [[[6,2], [243, 86]], [[280,2],[497,88]]],
        'Bike-Fall': [[[253,43],[451,176]],[[44, 274], [201,400]], [[264, 477], [451, 569]]],
        'Buff-Doge-vs-Cheems': [[[7,360],[297, 479]],[[377, 350], [647,495]]],
        'Clown-Applying-Makeup': [[[21,4], [440,183]], [[30,220], [439,373]], [[21, 412],[394,587]],[[21, 617],[443, 779]]],
        'Distracted-Boyfriend': [[[314, 10],[522, 215]],[[568, 3],[742, 138]],[[111, 301],[330, 491]]],
        'Drake': [[[612,15],[1181,578]],[[640,617],[1164,1163]]],
        'Hide-the-Pain-Harold': [[[24,17],[232,179]],[[14,335],[243,509]]],
        'Monkey-Puppet': [[[0,1],[288,160]],[[304,2],[580,174]]],
        'One-Does-Not-Simply': [[[83,11],[570,92]],[[138,240],[538,357]]],
        'Running-Away-Balloon': [[[331,46],[484,203]],[[7,399],[116,652]], [[202,424],[317,640]], [[390,386],[484,506]]],
        'Sleeping-Shaq': [[[9,6],[242,241]],[[17,24],[238,483]]],
        'Whisper-and-Goosebumps': [[[67,82],[560,286]]]
}

meme_cluster_mapping = {
        'Batman-Slapping-Robin': 2,
        'Bike-Fall': 3,
        'Buff-Doge-vs-Cheems': 2,
        'Clown-Applying-Makeup': 4,
        'Distracted-Boyfriend': 3,
        'Drake': 2,
        'Hide-the-Pain-Harold': 2,
        'Monkey-Puppet': 2,
        'One-Does-Not-Simply': 2,
        'Running-Away-Balloon': 4,
        'Sleeping-Shaq': 2,
        'Whisper-and-Goosebumps': 1
}

chat_string = ''
gpt_responses = ''
# Initialize Flask app


# Load your trained model
model_path = 'best_random_forest_model.joblib'
loaded_model = joblib.load(model_path)

# OpenAI setup
OPENAI_API_KEY = "sk-svcacct-6_-0FGRKPlBXDasvJYP7xRoOuRmX4tvunOrRmSd38r034bZbZXDF8wAlfs9etrT3BlbkFJYnHH0Iv1TaI6W1-5mKDmC04pdqh6_SAhxlKErd1oAIh27jPmh4qYWTJgpMMHYA"
client = OpenAI(api_key=OPENAI_API_KEY)

# Define routes for interaction
@app.route('/generate_meme_options', methods=['POST', 'OPTIONS'])
def generate_meme_options():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    # Handle the actual POST request
    data = request.json
    chat_history = data['chat_history']
    
    # Concatenate all chat messages into a single string in the format "user 1: message, user 2: message"
    concatenated_chat = ", ".join(f"{entry['userIndex']}: {entry['text']}" for entry in chat_history)
    app.logger.info(f"Concatenated chat history: {concatenated_chat}")
    chat_string = concatenated_chat
    gpt_response = analyze_conversation_to_meme(concatenated_chat)
    
    category, emotions, template, cur = construct_responses(gpt_response)
    probabilities = loaded_model.predict_proba(cur)
    top_2_predictions = [np.argsort(proba)[-2:][::-1] for proba in probabilities]
    top_2_results = [(proba[idx], loaded_model.classes_[idx]) for proba in probabilities for idx in np.argsort(proba)[-2:][::-1]]
    first_template = top_2_results[0][1]
    first_template = str(first_template)
    second_template = top_2_results[1][1]
    second_template = str(second_template)
    third_template = template_mapping[template]
    third_template = str(third_template)
    # Example response with meme options

    app.logger.info(f"GPT responses: {gpt_response}")
    
    meme_options = [
        first_template,
        second_template,
        third_template
    ]

    app.logger.info(f"Meme options: {meme_options}")

    response = {
        "meme_options": meme_options
    }
    return jsonify(response)




@app.route('/generate_final_meme', methods=['POST'])
def generate_final_meme():
    template_path = './templates/'
    data = request.json
    selected_meme = data['selected_meme']  # This is the template name without ".jpg"
    
    app.logger.info(f"Selected meme template: {selected_meme}")
    
    # Load the selected image path on the backend
    selected_image_path = os.path.join(template_path, f"{selected_meme}.jpg")
    if not os.path.exists(selected_image_path):
        return jsonify({"error": "Template image not found"}), 404

    # Define a cluster number (you may want to retrieve this from your logic or data)
    cluster_number = meme_cluster_mapping.get(selected_meme, 1)  # Default cluster as 1 if not found

    # Generate responses for the meme based on chat history or other data
    meme_responses = fit_to_meme(chat_string, selected_meme, cluster_number)
    proper_response = construct_and_manage_responses(meme_responses)

    # Open the template image
    output_img = add_text_to_meme(selected_image_path, selected_meme, proper_response)

    # Ensure image is in RGB mode for saving
    if output_img.mode == 'RGBA':
        output_img = output_img.convert('RGB')
    
    # Save the generated image
    output_file_path = os.path.join('./generated/', f"{selected_meme}_generated_meme.jpg")
    output_img.save(output_file_path)
    
    app.logger.info(f"Generated meme saved at: {output_file_path}")

    # Return the path to the generated meme
    return send_file(output_file_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
