from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer

# Flask app setup
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load a pretrained image classification model (e.g., CLIP or YOLO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load GPT-2 or GPT-Neo model from Hugging Face
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate captions from images using the classification model
def generate_captions(image_paths):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt")

    # Add a dummy text input to avoid errors
    dummy_text = ["a photo"] * len(image_paths)
    text_inputs = clip_processor(text=dummy_text, return_tensors="pt", padding=True)

    inputs.update(text_inputs)

    # Forward pass
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-to-text similarity scores

    # Generate captions based on similarity
    captions = []
    for logits in logits_per_image:
        top_caption_index = logits.argmax().item()
        captions.append(dummy_text[top_caption_index])

    return captions

# Generate story using GPT-2 model from Hugging Face
def generate_story_with_gpt2(captions):
    prompt = (
        f"Based on the following captions, generate an engaging and creative story for kids aged 5-12:\n\n"
        f"Captions: {', '.join(captions)}\n\n"
        f"The story should be fun, imaginative, and 200-500 words long."
    )
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text from GPT-2
    output = gpt2_model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
    story = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return story

# Process uploaded images and generate story
@app.route('/')
def index():
    return render_template('index.html', story=None, error=None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('index.html', story=None, error="No files uploaded.")

    files = request.files.getlist('file')

    if len(files) < 3 or len(files) > 10:
        return render_template('index.html', story=None, error="Please upload between 3 and 10 images.")

    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            return render_template('index.html', story=None, error="Invalid file format. Only images are allowed.")

    # Generate captions using the classification model
    captions = generate_captions(file_paths)

    # Generate story using GPT-2 based on object descriptions
    story = generate_story_with_gpt2(captions)

    return render_template('index.html', story=story, error=None)

if __name__ == '__main__':
    app.run(debug=True)
