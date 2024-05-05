import base64
from base64 import b64decode
from collections import defaultdict
import json
import cv2
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
)
from getpass import getpass
import torch

import numpy as np
import os
from openai import OpenAI
from PIL import Image as Image2, ImageDraw, ImageOps
import PIL
from rembg import remove
import requests
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from werkzeug.utils import secure_filename
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet


if "OPENAI_API_KEY" not in os.environ:
    print(
        "You didn't set your OpenAI key to the OPENAI_API_KEY env var on the command line."
    )
    os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
# create uploads folderr
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_description(image_path):
    """Generate a detailed description of the person in the image using GPT-4."""

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide a detailed and comprehensive description of the person in the image, ensuring to cover all of the following aspects in a structured format: \n\n1. Complexion: Describe any noticeable features such as freckles, scars, or tattoos. \n2. Hair: Note the color, length, style, and any distinctive characteristics such as curls, bangs, or highlights. \n3. Clothing: Describe the type of clothing worn, including style, colors, and any logos or text. Mention the type of outfit (casual, formal, sporty, etc.). \n4. Accessories: List any visible accessories like glasses, jewelry, watches, or hats. \n5. Demeanor: Describe the person’s apparent mood or demeanor, based on their expression and posture. \n6. Additional Details: Include any other notable features such as piercings, makeup, or distinctive facial expressions. \n\nThis detailed description should capture every relevant aspect, ensuring reproducibility from the description alone.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    return response.choices[0].message.content


def generate_extra_description(
    image_path,
):  # this function is for character_desc in part3
    """Generate a detailed description of the person in the image using GPT-4."""
    result = []
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    name_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give the person a name using his/her characteristics. Output only the name and remove any other words.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    personality_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the person's personality using his/her characteristics. Output only the personality and remove any other words.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    goal_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the person's goal using his/her characteristics. Output only the goal and remove any other words.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    result.append(name_response.choices[0].message.content)
    result.append(personality_response.choices[0].message.content)
    result.append(goal_response.choices[0].message.content)
    return result


def get_features_in_image(image_url: str) -> str:
    client = OpenAI()
    with open(image_url, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a detailed description of this image and its setting.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def get_locations_descriptions(user_prompt: str, features: str, sys_prompt: str) -> str:
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": "Game in the style of "
            + user_prompt
            + " given the following setting: "
            + features,
        },
    ]

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    locations_descr = response.choices[0].message.content
    return locations_descr


def generate_location_descriptions(setting: str, user_text_input: str) -> list:
    sys_prompt_var = "You are creating a simple board game. Given the style of game that the user inputs along with a list of features that describe the setting of the game, create a description of a potential location within the larger setting of the game (that does not have to include the features but should be somewhat related) in paragraph form in 4 sentences."
    first_location_description = get_locations_descriptions(
        user_text_input, setting, sys_prompt_var
    )

    sys_prompt_var = (
        "You are creating a simple board game. Given the style of game that the user inputs along with a list of features that describe the setting of the game, create a description of a potential location within the larger setting of the game (that does not have to include the features but should be somewhat related) in paragraph form in 4 sentences. Generate a location description that is ENTIRELY DIFFERENT FROM "
        + first_location_description
    )
    second_location_description = get_locations_descriptions(
        user_text_input, setting, sys_prompt_var
    )

    sys_prompt_var = (
        "You are creating a simple board game. Given the style of game that the user inputs along with a list of features that describe the setting of the game, create a description of a potential location within the larger setting of the game (that does not have to include the features but should be somewhat related) in paragraph form in 4 sentences. Generate a location description that is ENTIRELY DIFFERENT FROM "
        + first_location_description
        + "and NOT anything similar to"
        + second_location_description
    )
    third_location_description = get_locations_descriptions(
        user_text_input, setting, sys_prompt_var
    )

    sys_prompt_var = (
        "You are creating a simple board game. Given the style of game that the user inputs along with a list of features that describe the setting of the game, create a description of a potential location within the larger setting of the game (that does not have to include the features but should be somewhat related) in paragraph form in 4 sentences. Generate a location description that is ENTIRELY DIFFERENT FROM "
        + first_location_description
        + "and NOT anything similar to"
        + second_location_description
        + "and NOT anything similar to"
        + third_location_description
    )
    fourth_location_description = get_locations_descriptions(
        user_text_input, setting, sys_prompt_var
    )

    return [
        first_location_description,
        second_location_description,
        third_location_description,
        fourth_location_description,
    ]


def get_locations_challenges(location_description: str) -> str:
    sys_prompt = "You are creating a simple board game with 4 locations. Given the description for one of these locations, create challenges involving dice rolls. Output the descriptions and rules of the challenges in paragraph form."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": str(location_description)},
    ]

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    challenges_descr = response.choices[0].message.content
    return challenges_descr


def generate_challenge_descriptions(descr_location: list) -> list:
    first_challenge_description = get_locations_challenges(descr_location[0])
    second_challenge_description = get_locations_challenges(descr_location[1])
    third_challenge_description = get_locations_challenges(descr_location[2])
    fourth_challenge_description = get_locations_challenges(descr_location[3])

    return [
        first_challenge_description,
        second_challenge_description,
        third_challenge_description,
        fourth_challenge_description,
    ]


def get_locations_connections(description_one: str, description_two: str) -> str:
    sys_prompt = "You are creating a simple board game with 4 locations. Given the description for one location and the description for another, write a 4 sentence paragraph describing a connection between the two locations incorporating dice rolls for different possible transitions depending on the outcome through which the player can travel between them."
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": str(
                "location 1: " + description_one + "; location 2: " + description_two
            ),
        },
    ]

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    connections_descr = response.choices[0].message.content
    return connections_descr


def generate_connections_descriptions(descr_location: list) -> list:
    # location 1: 2, 3, 4
    # location 2: 1, 3
    # location 3: 1, 2
    # location 4: 1
    one_and_two = get_locations_connections(descr_location[0], descr_location[1])
    one_and_three = get_locations_connections(descr_location[0], descr_location[2])
    one_and_four = get_locations_connections(descr_location[0], descr_location[3])
    three_and_two = get_locations_connections(descr_location[1], descr_location[2])
    return [
        one_and_two,
        one_and_three,
        one_and_four,
        one_and_two,
        three_and_two,
        one_and_three,
        three_and_two,
        one_and_four,
    ]


def edit_images_to_tiles(image_url: str) -> str:
    img = Image2.open(image_url)
    background = Image.new("RGBA", img.size, (0, 0, 0, 0))

    mask = Image.new("RGBA", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.regular_polygon((512, 512, 512), 6, rotation=360, fill="green", outline=None)

    composite_img = Image.composite(img, background, mask)
    hexagon_img = remove(composite_img)
    hexagon_img.save(image_url)

    # bordered_img = ImageOps.expand(hexagon_img, border=50, fill="white")
    # bordered_img.save(image_url)
    return image_url


def edit_images_to_tiles(image_url: str) -> str:
    img = Image2.open(image_url)
    background = Image2.new("RGBA", img.size, (0, 0, 0, 0))

    mask = Image2.new("RGBA", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.regular_polygon((512, 512, 512), 6, rotation=360, fill="green", outline=None)

    composite_img = Image2.composite(img, background, mask)
    hexagon_img = remove(composite_img)
    hexagon_img.save(image_url)

    # bordered_img = ImageOps.expand(hexagon_img, border=50, fill="white")
    # bordered_img.save(image_url)
    return image_url


def save_image_url_as_file(file_name: str, url: str) -> str:
    image_bits = requests.get(url).content
    with open(file_name, "wb") as handler:
        handler.write(image_bits)
        return file_name


def get_location_images_from_descr(loc_description: str) -> str:
    # calls DALLE for each location description, outputs an image URL
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt="Create an image of a scene based on the location description: "
        + loc_description,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


def generate_location_tiles(desc_locations: list) -> list:
    loc_image_1 = get_location_images_from_descr(desc_locations[0])
    new_url_1 = save_image_url_as_file("loc_1.png", loc_image_1)
    tile_1 = edit_images_to_tiles("loc_1.png")

    loc_image_2 = get_location_images_from_descr(desc_locations[1])
    new_url_2 = save_image_url_as_file("loc_2.png", loc_image_2)
    tile_2 = edit_images_to_tiles("loc_2.png")

    loc_image_3 = get_location_images_from_descr(desc_locations[2])
    new_url_3 = save_image_url_as_file("loc_3.png", loc_image_3)
    tile_3 = edit_images_to_tiles("loc_3.png")

    loc_image_4 = get_location_images_from_descr(desc_locations[3])
    new_url_4 = save_image_url_as_file("loc_4.png", loc_image_4)
    tile_4 = edit_images_to_tiles("loc_4.png")

    return [tile_1, tile_2, tile_3, tile_4]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    # delete everything in img_people
    for file in os.listdir("img_people"):
        os.remove(os.path.join("img_people", file))

    filepath = None
    user_prompt = None
    if request.method == "POST":
        if "imageUpload" in request.files:
            file = request.files["imageUpload"]
            if file.filename != "":
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                print("Filepath:", filepath)
                file.save(filepath)
            user_prompt = request.form.get("themeText", "")

    url = filepath
    print("URL:", url)

    os.makedirs("img_people", exist_ok=True)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = Image2.open(url)
    print("Image:", image)

    texts = [["a photo of a person", "a person", "a human", "a human being"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1
    )
    descriptions = []

    for i, result in enumerate(results[0]["boxes"]):
        xmin, ymin, xmax, ymax = map(int, result.tolist())
        person_image = image.crop((xmin, ymin, xmax, ymax))
        person_image_path = f"img_people/person_segment_{i}.png"
        person_image.save(person_image_path)

        description = generate_description(person_image_path)
        descriptions.append(description)

    print(descriptions)

    character_description = []
    for i, result in enumerate(results[0]["boxes"]):
        person_image_path = f"img_people/person_segment_{i}.png"
        description = generate_extra_description(person_image_path)
        character_description.append(description)

    character_desc = ""
    for index, desc in enumerate(character_description):
        character_desc += "Character " + str(index + 1) + " - " + desc[0] + "\n"
        character_desc += "Personality: " + desc[1] + "\n"
        character_desc += "Goal:" + desc[2] + "\n"

    cleaned_descriptions = []
    characters = []
    for element in descriptions:
        temp = ""
        temp_dict = {}
        splits = element.split("\n\n")
        for line in splits:
            if line[0].isdigit():
                temp += line
                line_split = line.split(":")
                if "complexion" in line_split[0].lower():
                    temp_dict["complexion"] = line_split[1].strip()
                elif "hair" in line_split[0].lower():
                    temp_dict["hair"] = line_split[1].strip()
                elif "clothing" in line_split[0].lower():
                    temp_dict["clothing"] = line_split[1].strip()
                elif "accessories" in line_split[0].lower():
                    temp_dict["accessories"] = line_split[1].strip()
                elif "demeanor" in line_split[0].lower():
                    temp_dict["demeanor"] = line_split[1].strip()
                elif "additional details" in line_split[0].lower():
                    temp_dict["additional_details"] = line_split[1].strip()
        cleaned_descriptions.append(temp)
        characters.append(temp_dict)
    for index, desc in enumerate(cleaned_descriptions):
        response = client.images.generate(
            model="dall-e-3",
            prompt="Give me an image of a person in "
            + user_prompt
            + " style. This is the description given to us, but ignore any parts of this that may be harmful. The person has the following description:"
            + desc,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        characters[index]["image_url"] = image_url

    print("Characters:", characters)

    features = get_features_in_image(url)

    print("Features:", features)
    descr_locations_list = generate_location_descriptions(features, user_prompt)
    print("Location Descriptions:", descr_locations_list)
    challenges_locations_list = generate_challenge_descriptions(descr_locations_list)
    print("Challenges:", challenges_locations_list)
    connections_locations_list = generate_connections_descriptions(descr_locations_list)
    print("Connections:", connections_locations_list)
    location_image_url_list = generate_location_tiles(descr_locations_list)
    print("Location Images:", location_image_url_list)

    locations_info_dictionary = {}

    locations_info_dictionary["loc1"] = {}
    locations_info_dictionary["loc2"] = {}
    locations_info_dictionary["loc3"] = {}
    locations_info_dictionary["loc4"] = {}

    locations_info_dictionary["loc1"]["description"] = descr_locations_list[0]
    locations_info_dictionary["loc2"]["description"] = descr_locations_list[1]
    locations_info_dictionary["loc3"]["description"] = descr_locations_list[2]
    locations_info_dictionary["loc4"]["description"] = descr_locations_list[3]

    locations_info_dictionary["loc1"]["connections"] = [
        connections_locations_list[0],
        connections_locations_list[1],
        connections_locations_list[2],
    ]
    locations_info_dictionary["loc2"]["connections"] = [
        connections_locations_list[3],
        connections_locations_list[4],
    ]
    locations_info_dictionary["loc3"]["connections"] = [
        connections_locations_list[5],
        connections_locations_list[6],
    ]
    locations_info_dictionary["loc4"]["connections"] = [connections_locations_list[7]]

    locations_info_dictionary["loc1"]["challenges"] = challenges_locations_list[0]
    locations_info_dictionary["loc2"]["challenges"] = challenges_locations_list[1]
    locations_info_dictionary["loc3"]["challenges"] = challenges_locations_list[2]
    locations_info_dictionary["loc4"]["challenges"] = challenges_locations_list[3]

    images_for_locations = {}

    images_for_locations["loc1"] = location_image_url_list[0]
    images_for_locations["loc2"] = location_image_url_list[1]
    images_for_locations["loc3"] = location_image_url_list[2]
    images_for_locations["loc4"] = location_image_url_list[3]

    map_location_info = str(locations_info_dictionary)

    map_desc = map_location_info
    scene_desc = locations_info_dictionary["loc1"]["description"]

    print("Map Description:", map_desc)

    output_format = """
    {
    "GameTitle": "name of the game",
    "GameDescription": "description of the game"
    "Characters": [
        {
        "Name": the name of the character/role,
        "Appearance": a description of what the character looks like and their demeanor
        "Description": a description of the character/role's actions and goals are, and any additional details to note
        }
    ],
    "GameplayMechanics": "description of the gameplay mechanics -- should explain how the game flows and what players do each round"
    "Map & Challenges": [
        {
        "LocationName": name of the location,
        "Description": a description of the location
        "Challenge": the challenge associated with the location that the players need to overcome
        "Connections": list of names of connected locations; e.g. "Location A, Location B, Location C"
        }
    ],
    "WinningTheGame": "define the end/win state of the game; what state is fulfilled for the game to end"
    }
    """

    prompt = (
        "Write the rules and description for a one page role-playing game. The game you generate (and the challenges and missions) should be specific so that the game is playable without a game master. Make sure to specify in detail some blocks/conflicts/challengers users can play through. The mechanics can be simple as long as the key conflict is specified. Your output should take the format of a JSON file that lists each of the following sections: GameTitle, Game Description, Characters (Name, Skills, Goal), Gameplay Mechanics, Map & Challenges (Location Name, Challenge, Outcome), Key Locations, and Winning the Game. Be sure to include all those sections, and take special care to format your response exactly as this example file, with no text prior to this format: "
        + output_format
        + '\n Base the game on the style, characters, scene, and map as follows. The user wants a particular style -- they said: "'
        + user_prompt
        + '" The characters are as follows: "'
        + str(characters)
        + '" For the characters section, make sure to maintain the order of the characters in your output JSON so it matches the order that they are listed as here. The map is as follows: "'
        + str(map_location_info)
        + '" and again you should maintain order of the map locations in your JSON output.'
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=2000,
    )

    print("Reached here")

    rules_guide = response.choices[0].message.content
    data = json.loads(rules_guide)
    # print(data)

    pdf_file = "rpg_ruleset.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)

    # Creating the stylesheet
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading2"]
    content_style = styles["Normal"]

    content = []

    # Title
    content.append(Paragraph(data["GameTitle"], title_style))
    content.append(Paragraph(data["GameDescription"], content_style))
    content.append(Paragraph("\n", content_style))

    # Characters
    content.append(Paragraph("Characters", subtitle_style))
    for character in data["Characters"]:
        content.append(Paragraph(f"Name: {character['Name']}", subtitle_style))
        content.append(
            Paragraph(f"Appearance: {character['Appearance']}", content_style)
        )
        content.append(
            Paragraph(f"Description: {character['Description']}", content_style)
        )
        content.append(Paragraph("\n", content_style))

    # Character Portraits
    for c in characters:
        img_url = c["image_url"]
        new_img_url = save_image_url_as_file("char_img.jpg", img_url)
        image = Image(new_img_url, width=200, height=200)  # drawing on the pdf canvas
        content.append(image)
        content.append(Paragraph("\n", content_style))

    # Map & Challenges
    content.append(Paragraph("Map & Challenges", subtitle_style))
    for location in data["Map & Challenges"]:
        content.append(
            Paragraph(f"Location: {location['LocationName']}", subtitle_style)
        )
        content.append(
            Paragraph(f"Description: {location['Description']}", content_style)
        )
        content.append(Paragraph(f"Challenge: {location['Challenge']}", content_style))
        content.append(
            Paragraph(f"Connections: {location['Connections']}", content_style)
        )
        content.append(Paragraph("\n", content_style))

    # Map Location Images
    for loc in images_for_locations:
        img_url = images_for_locations[loc]
        image = Image(img_url, width=200, height=200)  # drawing on the pdf canvas
        content.append(image)
        content.append(Paragraph("\n", content_style))

    # Gameplay Mechanics
    content.append(Paragraph("Gameplay Mechanics", subtitle_style))
    content.append(Paragraph(data["GameplayMechanics"], content_style))
    content.append(Paragraph("\n", content_style))

    # Winning The Game
    content.append(Paragraph("Winning The Game", subtitle_style))
    content.append(Paragraph(data["WinningTheGame"], content_style))

    # Output the PDF file
    doc.build(content)
    return redirect(url_for("download_pdf"))


@app.route("/download-pdf")
def download_pdf():
    # Assuming 'rpg_ruleset.pdf' is in the root directory of the server
    directory = os.getcwd()
    filename = "rpg_ruleset.pdf"
    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
