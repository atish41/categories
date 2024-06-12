from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import PIL
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

app = Flask(__name__)

# Initialize Vertex AI
vertexai.init(project="travel-chatbot-409605", location="us-central1")

# System instruction for the model
system_instruction = """
You are an expert content categorizer at a social media company. Your job is to look at post images and their respective captions and derive categories related to them for better recommendations. 

Read the text and examine the image. Categorize each based on their content.

Categories:
- Determine the main theme or topic of the text.
- Identify the main subject and context of the image.

CATEGORIES_LIST={Gardening, Cooking and Baking, DIY and Crafts, Photography, Reading and Book Clubs, Gaming, Collecting (e.g., stamps, coins), Knitting and Sewing, Painting and Drawing, Running, Yoga and Pilates, Cycling, Hiking and Outdoor Activities, Team Sports (e.g., soccer, basketball), Swimming, Fitness and Bodybuilding, Martial Arts, Dance, Movies and TV Shows, Music and Concerts, Theater and Performing Arts, Comedy, Celebrity News and Gossip, Anime and Manga, Podcasts, Fan Clubs (e.g., specific bands, actors), Smartphones and Mobile Devices, Computers and Laptops, Smart Home Devices, Wearable Technology, Virtual Reality (VR) and Augmented Reality (AR), Gaming Consoles and Accessories, Software and Apps, Tech News and Reviews, Astronomy and Space, Biology and Medicine, Environmental Science, Physics and Chemistry, History and Archaeology, Mathematics, Language Learning, Educational Courses and Tutorials, Nutrition and Diet, Mental Health, Meditation and Mindfulness, Alternative Medicine, Fitness Challenges, Personal Development, Sleep and Relaxation, Wellness Retreats, Adventure Travel, Cultural Travel, Budget Travel, Luxury Travel, Road Trips, Travel Tips and Hacks, Travel Photography, Destination Reviews, Parenting, Dating and Relationships, Home Decor and Interior Design, Fashion and Style, Personal Finance, Minimalism, Eco-Friendly Living, Urban Living, Gourmet Cooking, Baking, Vegan and Vegetarian, Wine and Beer Tasting, Coffee Lovers, Food Photography, Restaurant Reviews, International Cuisine, Literature and Poetry, Visual Arts, Music and Instrumental, Theater and Performing Arts, Film and Documentary, Cultural Festivals, Art Exhibitions, Craftsmanship, Entrepreneurship, Freelancing, Networking, Career Development, Industry-Specific Groups (e.g., tech, finance), Job Hunting, Mentorship, Work-Life Balance, Environmental Activism, Human Rights, Animal Welfare, Political Activism, Community Service, Charitable Organizations, Sustainable Living, Diversity and Inclusion, Specific Fandoms (e.g., Harry Potter, Star Wars), Niche Collecting (e.g., rare books, vintage items), Unique Hobbies (e.g., urban beekeeping, rock balancing), Esoteric Interests (e.g., cryptozoology, paranormal), Startup Founders, Small Business Owners, Investment and Venture Capital, Business Strategy and Management, Marketing and Sales, E-commerce, Business Networking, Leadership and Mentoring, Home Renovation, Furniture Making, Landscaping and Gardening, DIY Home Decor, Plumbing and Electrical Projects, Sustainable Living Projects, Tool and Equipment Reviews, Upcycling and Recycling, Car Enthusiasts, Motorcycles, Electric Vehicles, Car Restoration, Off-Roading, Automotive News and Reviews, Motorsport, Vehicle Maintenance and Repair, Dog Owners, Cat Lovers, Exotic Pets, Animal Rescue and Adoption, Pet Training and Behavior, Pet Nutrition and Health, Aquariums and Fishkeeping, Bird Watching, Fiction Writing, Poetry, Non-Fiction Writing, Book Clubs, Literary Analysis, Writing Workshops, Publishing and Self-Publishing, Writing Prompts and Challenges, Goal Setting, Time Management, Productivity Hacks, Mindset and Motivation, Public Speaking, Journaling, Coaching and Mentoring, Life Skills, Skincare and Makeup, Fashion Trends, Personal Styling, Beauty Tutorials, Sustainable Fashion, Haircare, Nail Art, Fashion Design, Meditation and Mindfulness, Yoga and Spiritual Practices, Religious Study Groups, Comparative Religion, Spiritual Growth, Astrology and Horoscopes, Spiritual Healing, Rituals and Ceremonies, Web Development, Mobile App Development, Data Science and Machine Learning, Cybersecurity, Cloud Computing, Software Engineering, Programming Languages, Hackathons and Coding Challenges, Historical Events, Archaeology, Genealogy, Cultural Studies, Historical Reenactments, Ancient Civilizations, Military History, Preservation and Restoration, Renewable Energy, Zero Waste Lifestyle, Sustainable Agriculture, Green Building, Environmental Policy, Eco-Friendly Products, Climate Change Action, Conservation Efforts, Stock Market, Cryptocurrency, Real Estate Investment, Personal Finance Management, Retirement Planning, Budgeting and Saving, Financial Independence, Investment Strategies, New Parents, Single Parenting, Parenting Teens, Child Development, Educational Resources for Kids, Work-Life Balance for Parents, Parenting Support Groups, Family Activities and Outings, Language Learning (e.g., Spanish, French, Mandarin), Cultural Exchange, Translation and Interpretation, Linguistics, Language Immersion Programs, Dialects and Regional Languages, Multilingual Communities, Language Teaching Resources, Mental Health Awareness, Physical Fitness Challenges, Holistic Health, Sports Psychology, Body Positivity, Mind-Body Connection, Stress Management, Chronic Illness Support, Camping and Backpacking, Bird Watching, Nature Photography, Rock Climbing, Fishing and Hunting, Wildcrafting and Foraging, Stargazing, National Parks Exploration, Pottery and Ceramics, Jewelry Making, Scrapbooking, Candle Making, Textile Arts, Glass Blowing, Woodworking, Paper Crafts, Independent Filmmaking, Screenwriting, Animation and VFX, Documentary Filmmaking, Video Editing, Cinematography, Media Critique and Analysis, Podcast Production}
Return the categories as terms(single words as given in the list) separated by commas. DON'T RETURN FULL SENTENCES.
Return the categories as terms separated by commas in a list. 
return results in one list
"""

# Generation configuration for the model
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}

# Initialize the model
model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])

def fetch_and_preprocess_image(image_url):
    """Fetches the image from the URL, converts it to a format suitable for the model, and returns a Part object."""
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224), PIL.Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return Part.from_data(mime_type="image/jpeg", data=image_bytes)

def predict_categories(caption, image_url):
    """Combines caption and image into parts, calls the model, and returns the predicted categories."""
    caption_part = Part.from_text(caption)
    image_part = fetch_and_preprocess_image(image_url)
    contents = [caption_part, image_part]
    response = model.generate_content(contents, generation_config=generation_config)
    return response.text.strip()  # Remove leading/trailing whitespace

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'caption' not in data or 'image_url' not in data:
        return jsonify({"error": "Please provide both 'caption' and 'image_url' in the request body"}), 400

    caption = data['caption']
    image_url = data['image_url']

    try:
        predicted_categories = predict_categories(caption, image_url)
        return jsonify({"categories": predicted_categories.split(",")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

\
if __name__ == '__main__':
    app.run(debug=True,port=4000)
