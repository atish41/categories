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
Return the categories as terms separated by commas in a list.
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])

def fetch_and_preprocess_image(image_path_or_file):
    if isinstance(image_path_or_file, str):
        if image_path_or_file.startswith("http://") or image_path_or_file.startswith("https://"):
            response = requests.get(image_path_or_file)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_file)
    else:
        image = Image.open(image_path_or_file)

    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return Part.from_data(mime_type="image/jpeg", data=image_bytes)

def fetch_and_preprocess_video(video_path_or_file):
    temp_file_path = None
    try:
        if isinstance(video_path_or_file, str):
            if video_path_or_file.startswith("http://") or video_path_or_file.startswith("https://"):
                response = requests.get(video_path_or_file, stream=True)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                temp_file_path = temp_file.name
            else:
                temp_file_path = video_path_or_file
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            temp_file.write(video_path_or_file.read())
            temp_file.close()
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path_or_file}")

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
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return frames

def predict_categories(caption, image_paths_or_files, video_paths_or_files):
    contents = [Part.from_text(caption)]
    unique_categories = set()

    # Process the text caption separately
    response = model.generate_content(contents, generation_config=generation_config)
    categories = response.text.strip().split(", ")
    unique_categories.update(categories)

    for image_path_or_file in image_paths_or_files:
        image_part = fetch_and_preprocess_image(image_path_or_file)
        content = contents + [image_part]
        response = model.generate_content(content, generation_config=generation_config)
        categories = response.text.strip().split(", ")
        unique_categories.update(categories)

    for video_path_or_file in video_paths_or_files:
        video_frames = fetch_and_preprocess_video(video_path_or_file)
        for frame in video_frames:
            content = contents + [frame]
            response = model.generate_content(content, generation_config=generation_config)
            categories = response.text.strip().split(", ")
            unique_categories.update(categories)

    return list(unique_categories)

app = Flask(__name__)

@app.route('/predict_categories', methods=['POST'])
def predict():
    caption = request.form.get('caption', '').strip()
    image_files = request.files.getlist('images')
    video_files = request.files.getlist('videos')

    if not caption or (not image_files and not video_files):
        return jsonify({"error": "Please provide a caption and at least one image or video file."}), 400

    try:
        predicted_categories = predict_categories(caption, image_files, video_files)
        return jsonify({"predicted_categories": predicted_categories}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7000)
    

"""In this code:
- The `fetch_and_preprocess_image` and `fetch_and_preprocess_video` functions are updated to handle file-like objects.
- The `predict_categories` function now accepts a list of file-like objects for images and videos.
- The Flask endpoint `/predict_categories` uses `request.form` to get the caption and `request.files` to get the uploaded files.

### Step 2: Test with Postman

1. **Set Up Postman**:
    - Open Postman.
    - Create a new POST request to `http://localhost:5000/predict_categories`.
    - In the `Headers` tab, ensure that `Content-Type` is set to `multipart/form-data`.

2. **Add Form Data**:
    - Select the `Body` tab.
    - Choose `form-data`.
    - Add a key for `caption` with your text caption.
    - Add keys for `images` and `videos` for each file you want to upload.
        - For images and videos, use the `Choose Files` option to select files from your local machine.
        - Ensure each file key is named `images` for image files and `videos` for video files.

3. **Send the Request**:
    - Click `Send`.
    - Check the response to see the predicted categories.

This setup ensures the Flask app can handle both URLs and local file uploads, making it versatile for various use cases."""
