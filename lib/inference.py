from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import re
something="sk-svcacct-6_-0FGRKPlBXDasvJYP7xRoOuRmX4tvunOrRmSd38r034bZbZXDF8wAlfs9etrT3BlbkFJYnHH0Iv1TaI6W1-5mKDmC04pdqh6_SAhxlKErd1oAIh27jPmh4qYWTJgpMMHYA"
client = OpenAI(api_key = something)

def analyze_conversation_to_meme(text):
    prompt = f"""
        1. Determine the category of this meme this text could belong to:
        
        1. Memes: internet meme documentation.
        2. Events: specific internet event documentation.
        3. People: covers biographical info & highlights of individuals or collectives.
        4. Sites: covers notable internet hubs or online communities.
        5. Subcultures: particular interests or activities tied to certain groups.
        
        2. Analyze the following conversation and determine if it contains emotions that belong to these categories:
        
        1. Amusement/Enjoyment
        2. Sarcasm/Irony
        3. Frustration/Annoyance
        4. Nostalgia
        5. Self-Deprecation
        6. Confusion/Surprise
        7. Cynicism/Pessimism
        8. Wholesomeness/Pure Joy
        9. Empathy/Relatability
        10. Mock Pride/Confidence

        3. Which meme template suit it the most?
        1.'Batman-Slapping-Robin',
        2.'Bike-Fall',
        3.'Buff-Doge-vs-Cheems',
        4.'Clown-Applying-Makeup',
        5.'Distracted-Boyfriend',
        6.'Drake',
        7.'Hide-the-Pain-Harold',
        8.'Monkey-Puppet',
        9.'One-Does-Not-Simply',
        10.'Running-Away-Balloon',
        11.'Sleeping-Shaq',
        12.'Whisper-and-Goosebumps'

        
        
        Conversation:
        {text}
        
        Return the answers in the exact format: 'Category: 1; Emotions: 2, 5; Template: 1'.
        """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    transition_analysis = response.choices[0].message.content
    return transition_analysis


def construct_responses(response):
    column_names = ['Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5',
       '0', '1', '2']
    cur = pd.DataFrame(columns = column_names)
    response_list = response.split(";")
    result = []
    for i in response_list:
        i = i.split(":")[1]
        result.append(i)
    category = int(re.search(r'\d+', result[0]).group())
    emotions = result[1].split(",")
    emotions = list(map(lambda s: int(re.search(r'\d+', s).group()), emotions))
    template = int(re.search(r'\d+', result[2]).group())

    default_list = [False, False, False, False, False, 0, 0, 0]
    for i in range (len(default_list)):
        if i+1 == category:
            default_list[i] = True
        if i>4 :
            try:
                default_list[i] = emotions[i-5]
            except:
                continue
    cur.loc[len(cur)] = default_list           
    return category, emotions, template, cur
    
def fit_to_meme(text, template, cluster_number):
    prompt = f"""
    1. Make the following conversation several very very very short meme style parts that can be suitable for meme:
    Conversation:
    {text}
    Meme template:
    {template}
    How many parts:
    {cluster_number}
    
    Return the answers in the exact format and don't mention who is talking, remember the part number should match the {cluster_number}: Part1: ; Part2: ; ....
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    transition_analysis = response.choices[0].message.content
    return transition_analysis

def construct_and_manage_responses(responses):
    response_list = responses.split('\n')
    return_list = []
    for i in response_list:
        c_r = i.split(':')[-1]
        return_list.append(c_r)

    return return_list


from PIL import Image, ImageDraw, ImageFont
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

def add_text_to_meme(image_path, template_name, texts):
    """
    Draws text within specified bounding boxes on the meme template.
    
    Parameters:
    - image_path (str): Path to the image.
    - template_name (str): The name of the template to get bounding boxes from meme_location_mapping.
    - texts (list): List of text strings for each bounding box on the template.
    
    Returns:
    - Image with text drawn within bounding boxes.
    """
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Get bounding boxes for the template
    boxes = meme_location_mapping.get(template_name, [])
    
    for i, box in enumerate(boxes):
        if i < len(texts):
            # Unpack the bounding box coordinates
            (x1, y1), (x2, y2) = box
            text = texts[i]
            
            # Calculate the maximum width and height for the text
            max_width = x2 - x1
            max_height = y2 - y1
            
            # Adjust the font size dynamically to fit text within the bounding box
            font = adjust_font_size(text, max_width, max_height)
            
            # Wrap text based on the bounding box width and height
            wrapped_text = wrap_text_to_box(text, font, max_width, max_height)

            # Draw the text, centered vertically and horizontally within the box
            text_y = y1 + (max_height - len(wrapped_text) * font.getbbox(' ')[3]) // 2
            for line in wrapped_text:
                text_width = font.getbbox(line)[2] - font.getbbox(line)[0]
                text_x = x1 + (max_width - text_width) // 2
                draw.text((text_x, text_y), line, fill="black", font=font)
                text_y += font.getbbox(' ')[3]

    return img

def adjust_font_size(text, max_width, max_height, start_size=60):
    """
    Adjusts font size to fit the text within the specified width and height.
    
    Parameters:
    - text (str): Text to fit within the bounding box.
    - max_width (int): Maximum width of the bounding box.
    - max_height (int): Maximum height of the bounding box.
    - start_size (int): Starting font size.
    
    Returns:
    - ImageFont.FreeTypeFont object with the adjusted font size.
    """
    font_size = start_size
    font = ImageFont.truetype("arial.ttf", font_size)  # Use a scalable font, e.g., Arial
    
    while font.getbbox(text)[2] > max_width or font.getbbox(text)[3] * len(text.split()) > max_height:
        font_size -= 1
        font = ImageFont.truetype("arial.ttf", font_size)
        if font_size < 10:  # Break if the font gets too small
            break
            
    return font




def wrap_text_to_box(text, font, max_width, max_height):
    """
    Wraps text to fit within a specified width and height by adjusting line breaks.
    
    Parameters:
    - text (str): The text to wrap.
    - font (ImageFont): The font used to measure text size.
    - max_width (int): The maximum width of the box.
    - max_height (int): The maximum height of the box.
    
    Returns:
    - List of strings where each string is a line of wrapped text.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word would exceed the max width
        test_line = f"{current_line} {word}".strip()
        if font.getbbox(test_line)[2] <= max_width:
            current_line = test_line
        else:
            # Add the current line to lines and reset
            lines.append(current_line)
            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Limit lines to fit within max_height
    line_height = font.getbbox('A')[3]
    max_lines = max_height // line_height
    return lines[:max_lines]




