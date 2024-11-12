from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
from openai import OpenAI
from lib import *
import json  # Import the json module
import pandas
from datetime import datetime
import psycopg2
import random

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
        12: 'Whisper-and-Goosebumps',
        13: 'Anakin-Padme-4-Panel',
        14: 'Bernie-I-Am-Once-Again-Asking-For-Your-Support',
        15: 'Boardroom-Meeting-Suggestion',
        16: 'Expanding-Brain',
        17: 'Guess-Ill-die',
        18: 'I-Bet-Hes-Thinking-About-Other-Women',
        19: 'Is-This-A-Pigeon',
        20: 'Laughing-Leo',
        21: 'Leonardo-Dicaprio-Cheers',
        22: 'Roll-Safe-Think-About-It',
        23: 'Sad-Pablo-Escobar',
        24: 'Success-Kid',
        25: 'Surprised-Pikachu',
        26: 'The-Rock-Driving',
        27: 'This-Is-Fine',
        28: 'Tuxedo-Winnie-The-Pooh',
        29: 'Two-Buttons',
        30: 'Waiting-Skeleton',
        31: 'Woman-Yelling-At-Cat'
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
        'Whisper-and-Goosebumps': [[[67,82],[560,286]]],
        'Anakin-Padme-4-Panel':[[[5, 220], [365,368]], [[436,227],[742,364]], [[17,521], [366,728]], [[454, 571], [691, 710]]],
        'Bernie-I-Am-Once-Again-Asking-For-Your-Support': [[[164,655], [601,721]]],
        'Boardroom-Meeting-Suggestion':[[[154, 8], [440, 41]], [[30,248], [116,269]], [[164, 256], [240, 275]]],
        'Expanding-Brain': [[[9,11],[408,280]],[[18,326],[395,573]], [[5,620], [412, 857]], [[11,911], [404,1178]]],
        'Guess-Ill-die': [[[12,8], [376,124]], [[49,224],[343,292]]],
        'I-Bet-Hes-Thinking-About-Other-Women': [[[52, 118], [773,414]],[[835,220],[1649,605]]],
        'Is-This-A-Pigeon': [[[34,330],[708,774]],[[929,229],[1576,597]],[[122,949], [1504, 1275]]],
        'Laughing-Leo': [[[32,11],[443,174]], [[13,342], [450,445]]],
        'Leonardo-Dicaprio-Cheers': [[[20,9],[590,107]],[[12,286],[587,379]]],
        'Roll-Safe-Think-About-It': [[[23,10], [686,135]], [[43,275], [651, 374]]],
        'Sad-Pablo-Escobar': [[[56,11],[654,300]],[[26,438], [341,603]], [[377,461], [694,637]]],
        'Success-Kid': [[[19,9],[492, 165]], [[17,355], [483, 477]]],
        'Surprised-Pikachu': [[[24,25],[1864,728]]],
        'The-Rock-Driving': [[[331, 35],[549,139]], [[334,265], [556,362]]],
        'This-Is-Fine': [[[15, 14], [278,103]]],
        'Tuxedo-Winnie-The-Pooh': [[[350,4],[791,286]], [[351, 301], [791,566]]],
        'Two-Buttons': [[[37,92], [273, 170]], [[266,65], [432, 116]], [[86,734], [553, 839]]],
        'Waiting-Skeleton':[[[13, 5], [294, 109]], [[9, 275], [294, 365]]],
        'Woman-Yelling-At-Cat': [[[1, 1], [322,85]], [[350, 4], [670,90]]]
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
        'Whisper-and-Goosebumps': 1,
        'Anakin-Padme-4-Panel': 4,
        'Bernie-I-Am-Once-Again-Asking-For-Your-Support': 1,
        'Boardroom-Meeting-Suggestion': 4,
        'Expanding-Brain': 4,
        'Guess-Ill-die': 2,
        'I-Bet-Hes-Thinking-About-Other-Women': 2,
        'Is-This-A-Pigeon': 3,
        'Laughing-Leo': 2,
        'Leonardo-Dicaprio-Cheers': 2,
        'Roll-Safe-Think-About-It': 2,
        'Sad-Pablo-Escobar': 3,
        'Success-Kid': 2,
        'Surprised-Pikachu': 1,
        'The-Rock-Driving': 2,
        'This-Is-Fine': 1,
        'Tuxedo-Winnie-The-Pooh': 2,
        'Two-Buttons': 2,
        'Waiting-Skeleton': 2,
        'Woman-Yelling-At-Cat': 2
}

chat_string = ''
# Initialize Flask app


# Load your trained model
model_path = 'best_random_forest_model.joblib'
loaded_model = joblib.load(model_path)
#DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'saved_data', 'chat_data.csv')
# OpenAI setup

# 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

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

DATABASE_URL = os.getenv('DATABASE_URL')

# Function to connect to the database and create the table if it doesn’t exist
def create_table():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_data (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255),
            chat_history TEXT,
            num_users INTEGER,
            selected_meme_index INTEGER,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def validate_data(data):
    required_fields = ['username', 'chatHistory', 'numUsers', 'selectedMemeIndex']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")
        if field in ['numUsers', 'selectedMemeIndex'] and not isinstance(data[field], int):
            raise ValueError(f"{field} must be an integer")


def preprocess_data(data):
    # Convert any nested dicts or lists to strings
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            data[key] = str(value)  # Convert to JSON string representation
    return data

def format_chat_history(chat_history):
    formatted_text = []
    for entry in chat_history:
        user_index = entry.get('userIndex')
        text = entry.get('text')
        formatted_text.append(f"userIndex: {user_index}, text: {text}")
    return " | ".join(formatted_text)  # Concatenate with a separator like " | " if needed



def save_data_to_postgres(data):
    user_name = data.get('username')
    chatHistory = format_chat_history(data.get('chatHistory'))
    numUsers = data.get('numUsers')
    selectedMemeIndex = data.get('selectedMemeIndex')
    try:
        # Add timestamp to the data
        #data['timestamp'] = datetime.now().isoformat()

        # Connect to the PostgreSQL database
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cursor = conn.cursor()

        # Insert data into the table
        insert_query = '''
            INSERT INTO chat_data (username, chat_history, num_users, selected_meme_index)
            VALUES (%s, %s, %s, %s)
        '''
        cursor.execute(insert_query, (
            user_name,
            chatHistory,
            numUsers,
            selectedMemeIndex
        ))


        # Commit the transaction
        conn.commit()
        print("Data successfully saved to PostgreSQL.")
        
        # Close the connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Failed to save data to PostgreSQL: {e}")
        raise


# Call the create_table function when the app starts to ensure the table exists
create_table()
        
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
    # second_template = top_2_results[1][1]
    # second_template = str(second_template)
    third_template = template_mapping[template]

    

    third_template = str(third_template)

    excluded_templates = {first_template, third_template}
    
    random_templates = [key for key in template_mapping.keys() if key not in excluded_templates]
    random_template = random.choice(random_templates)
    second_template = str(random_template)
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
    chat_history = data['chat_history']
    concatenated_chat = ", ".join(f"{entry['userIndex']}: {entry['text']}" for entry in chat_history)
    

    app.logger.info(f"chat history: {concatenated_chat}")
    meme_responses = fit_to_meme(concatenated_chat, selected_meme, cluster_number)
    
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



@app.route('/save_chat_data', methods=['POST'])
def save_chat_data():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    data = request.json  # Expecting a JSON payload from the client
    app.logger.info(f"Received data to save: {data}")

    # Print data structure for debugging
    print("Data received:", data)
    print("Data type:", type(data))

    # Store data in PostgreSQL
    try:
        validate_data(data)  # Validate input data
        save_data_to_postgres(data)
        return jsonify({"status": "success", "message": "Data saved successfully"}), 200
    except ValueError as ve:
        app.logger.error(f"Validation error: {ve}")
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Failed to save data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500






if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug = True)
