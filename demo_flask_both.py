import os
import tempfile
from io import BytesIO
from flask import Flask, request, jsonify
import requests
from PIL import Image
import cv2
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Initialize Vertex AI
vertexai.init(project="travel-chatbot-409605", location="us-central1")

system_instruction = """
You are an expert content categorizer at a social media company. Your job is to look at post images and their respective captions and derive categories related to them for better recommendations. 

Read the text and examine the image. Categorize each based on their content.

Categories:
- Determine the main theme or topic of the text.
- Identify the main subject and context of the image.
- Identify the main subject and context of the video.

CATEGORIES_LIST={Gardening, Cooking and Baking, DIY and Crafts, Photography, Reading and Book Clubs, Gaming, Collecting (e.g., stamps, coins), Knitting and Sewing, Painting and Drawing, Running, Yoga and Pilates, Cycling, Hiking and Outdoor Activities, Team Sports (e.g., soccer, basketball), Swimming, Fitness and Bodybuilding, Martial Arts, Dance, Movies and TV Shows, Music and Concerts, Theater and Performing Arts, Comedy, Celebrity News and Gossip, Anime and Manga, Podcasts, Fan Clubs (e.g., specific bands, actors), Smartphones and Mobile Devices, Computers and Laptops, Smart Home Devices, Wearable Technology, Virtual Reality (VR) and Augmented Reality (AR), Gaming Consoles and Accessories, Software and Apps, Tech News and Reviews, Astronomy and Space, Biology and Medicine, Environmental Science, Physics and Chemistry, History and Archaeology, Mathematics, Language Learning, Educational Courses and Tutorials, Nutrition and Diet, Mental Health, Meditation and Mindfulness, Alternative Medicine, Fitness Challenges, Personal Development, Sleep and Relaxation, Wellness Retreats, Adventure Travel, Cultural Travel, Budget Travel, Luxury Travel, Road Trips, Travel Tips and Hacks, Travel Photography, Destination Reviews, Parenting, Dating and Relationships, Home Decor and Interior Design, Fashion and Style, Personal Finance, Minimalism, Eco-Friendly Living, Urban Living, Gourmet Cooking, Baking, Vegan and Vegetarian, Wine and Beer Tasting, Coffee Lovers, Food Photography, Restaurant Reviews, International Cuisine, Literature and Poetry, Visual Arts, Music and Instrumental, Theater and Performing Arts, Film and Documentary, Cultural Festivals, Art Exhibitions, Craftsmanship, Entrepreneurship, Freelancing, Networking, Career Development, Industry-Specific Groups (e.g., tech, finance), Job Hunting, Mentorship, Work-Life Balance, Environmental Activism, Human Rights, Animal Welfare, Political Activism, Community Service, Charitable Organizations, Sustainable Living, Diversity and Inclusion, Specific Fandoms (e.g., Harry Potter, Star Wars), Niche Collecting (e.g., rare books, vintage items), Unique Hobbies (e.g., urban beekeeping, rock balancing), Esoteric Interests (e.g., cryptozoology, paranormal), Startup Founders, Small Business Owners, Investment and Venture Capital, Business Strategy and Management, Marketing and Sales, E-commerce, Business Networking, Leadership and Mentoring, Home Renovation, Furniture Making, Landscaping and Gardening, DIY Home Decor, Plumbing and Electrical Projects, Sustainable Living Projects, Tool and Equipment Reviews, Upcycling and Recycling, Car Enthusiasts, Motorcycles, Electric Vehicles, Car Restoration, Off-Roading, Automotive News and Reviews, Motorsport, Vehicle Maintenance and Repair, Dog Owners, Cat Lovers, Exotic Pets, Animal Rescue and Adoption, Pet Training and Behavior, Pet Nutrition and Health, Aquariums and Fishkeeping, Bird Watching, Fiction Writing, Poetry, Non-Fiction Writing, Book Clubs, Literary Analysis, Writing Workshops, Publishing and Self-Publishing, Writing Prompts and Challenges, Goal Setting, Time Management, Productivity Hacks, Mindset and Motivation, Public Speaking, Journaling, Coaching and Mentoring, Life Skills, Skincare and Makeup, Fashion Trends, Personal Styling, Beauty Tutorials, Sustainable Fashion, Haircare, Nail Art, Fashion Design, Meditation and Mindfulness, Yoga and Spiritual Practices, Religious Study Groups, Comparative Religion, Spiritual Growth, Astrology and Horoscopes, Spiritual Healing, Rituals and Ceremonies, Web Development, Mobile App Development, Data Science and Machine Learning, Cybersecurity, Cloud Computing, Software Engineering, Programming Languages, Hackathons and Coding Challenges, Historical Events, Archaeology, Genealogy, Cultural Studies, Historical Reenactments, Ancient Civilizations, Military History, Preservation and Restoration, Renewable Energy, Zero Waste Lifestyle, Sustainable Agriculture, Green Building, Environmental Policy, Eco-Friendly Products, Climate Change Action, Conservation Efforts, Stock Market, Cryptocurrency, Real Estate Investment, Personal Finance Management, Retirement Planning, Budgeting and Saving, Financial Independence, Investment Strategies, New Parents, Single Parenting, Parenting Teens, Child Development, Educational Resources for Kids, Work-Life Balance for Parents, Parenting Support Groups, Family Activities and Outings, Language Learning (e.g., Spanish, French, Mandarin), Cultural Exchange, Translation and Interpretation, Linguistics, Language Immersion Programs, Dialects and Regional Languages, Multilingual Communities, Language Teaching Resources, Mental Health Awareness, Physical Fitness Challenges, Holistic Health, Sports Psychology, Body Positivity, Mind-Body Connection, Stress Management, Chronic Illness Support, Camping and Backpacking, Bird Watching, Nature Photography, Rock Climbing, Fishing and Hunting, Wildcrafting and Foraging, Stargazing, National Parks Exploration, Pottery and Ceramics, Jewelry Making, Scrapbooking, Candle Making, Textile Arts, Glass Blowing, Woodworking, Paper Crafts, Independent Filmmaking, Screenwriting, Animation and VFX, Documentary Filmmaking, Video Editing, Cinematography, Media Critique and Analysis, Podcast Production}
Return the categories as terms(single words as given in the list) separated by commas. DON'T RETURN FULL SENTENCES.
Return the categories as terms separated by commas in a list. 
return results in one list
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])

def fetch_and_preprocess_image(image_path):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return Part.from_data(mime_type="image/jpeg", data=image_bytes)

def fetch_and_preprocess_video(video_path):
    temp_file_path = None
    try:
        if video_path.startswith("http://") or video_path.startswith("https://"):
            response = requests.get(video_path, stream=True)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        success, frame = cap.read()
        frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 5)

        while success and len(frames) < 5:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
            success, frame = cap.read()

        cap.release()
    finally:
        if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return frames

def predict_categories(caption, image_paths, video_paths):
    contents = [Part.from_text(caption)]
    unique_categories = set()

    # Process the text caption separately
    response = model.generate_content(contents, generation_config=generation_config)
    categories = response.text.strip().split(", ")
    unique_categories.update(categories)

    for image_path in image_paths:
        image_part = fetch_and_preprocess_image(image_path)
        content = contents + [image_part]
        response = model.generate_content(content, generation_config=generation_config)
        categories = response.text.strip().split(", ")
        unique_categories.update(categories)

    for video_path in video_paths:
        video_frames = fetch_and_preprocess_video(video_path)
        for frame in video_frames:
            content = contents + [frame]
            response = model.generate_content(content, generation_config=generation_config)
            categories = response.text.strip().split(", ")
            unique_categories.update(categories)

    return list(unique_categories)

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def predict():
    data = None

    # Handle different content types
    if request.content_type == 'application/json':
        data = request.json
    elif request.content_type.startswith('multipart/form-data'):
        data = {
            'caption': request.form.get('caption', '').strip(),
            'image_urls': request.form.getlist('image_urls'),
            'video_urls': request.form.getlist('video_urls')
        }

    if not data:
        return jsonify({"error": "Unsupported content type or missing data"}), 400

    caption = data.get('caption', '').strip()
    image_paths = [url.strip() for url in data.get('image_urls', []) if url.strip()]
    video_paths = [url.strip() for url in data.get('video_urls', []) if url.strip()]

    if not caption or (not image_paths and not video_paths):
        return jsonify({"error": "Please provide a caption and at least one image or video URL."}), 400

    try:
        predicted_categories = predict_categories(caption, image_paths, video_paths)
        return jsonify({"predicted_categories": predicted_categories}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9000)
