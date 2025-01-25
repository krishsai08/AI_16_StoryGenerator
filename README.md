
# **Project Name: AI-Powered Story Generator for Kids Using Images as Input**

## **a. Project Overview**

### **Title**: Story Generator using Object Detection and GPT-2

### **Problem Statement Addressed**:
This project aims to generate creative and engaging stories based on the objects detected in uploaded images. The primary goal is to enable users to upload multiple images, detect objects within those images, and then use the object descriptions to automatically generate a story using a language model (GPT-2). This addresses the need for interactive, creative content generation in the fields of education, entertainment, and content creation.

### **Objective and Goals**:
The key objectives of the project are:
- To integrate computer vision with NLP models to generate stories.
- To allow users to upload a series of images, detect objects, and create stories that make sense based on those objects.
- To use a combination of YOLOv8 for object detection and GPT-2 for text generation.

### **Key Features and Functionalities**:
- **Image Uploading**: Users can upload between 3 to 10 images.
- **Object Detection**: YOLOv8 is used to detect and classify objects in the images.
- **Story Generation**: GPT-2 uses the detected objects to create a short, engaging story.
- **Web Interface**: A Flask-based web application is used to handle image uploads and story display.

### **Target Audience or Use Case**:
- **Students and Educators**: For creative storytelling and educational purposes.
- **Content Creators**: For generating ideas and stories from image-based inputs.
- **Developers**: For exploring the integration of AI models in web applications.

---

## **b. Technology Stack**

- **Programming Languages**: Python 3.x
- **Libraries/Frameworks**:
  - **Flask**: For the web framework.
  - **OpenCV**: For image processing and displaying results.
  - **YOLOv8**: For object detection.
  - **HuggingFace Transformers**: For text generation using GPT-2.
  - **Torch**: For model inference on GPU.
  - **Werkzeug**: For handling file uploads in Flask.

- **Tools and Platforms**:
  - **YOLOv8** (Object Detection)
  - **GPT-2** (Text Generation via HuggingFace)
  - **Google Colab** (for model development and training)

---

## **c. System Requirements**

### **Hardware Requirements**:
- Minimum: 4 GB RAM (recommended 8 GB)
- GPU with CUDA support (for faster object detection and model inference). However, CPU can also be used with slower performance.

### **Software Requirements**:
- Python 3.8 or higher
- Pip (Python package installer)
- Flask
- OpenCV
- PyTorch
- HuggingFace Transformers
- YOLOv8 pre-trained model

---

## **d. Installation and Setup Instructions**

### **Step-by-Step Setup**:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/krishsai08/TeamAI_16_StoryGenerator.git
   cd TeamAI_16_StoryGenerator
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Configuration** (If applicable):
   - There is no database required for this project, as it processes the images and generates a story dynamically.

5. **Run the Flask application**:
   ```bash
   python app.py
   ```

   The app will be available at `http://127.0.0.1:5000/`.

### **Screenshots**:
- **Image Upload**: Show a screenshot of the upload page.
- **Result Page**: Show a screenshot of a generated story.

---

## **e. Features Explanation**

- **Image Upload**: Users can select and upload between 3 to 10 images. The images will be processed for object detection.
- **Object Detection**: The YOLOv8 model identifies objects in the uploaded images, such as animals, furniture, etc.
- **Story Generation**: The identified objects are passed to GPT-2, which generates a creative story based on the detected objects.

### **GIF/Video Demos**:
- Demonstrate the upload process and generated story output.

---

## **f. Usage Instructions**

### **For Job Seekers and Employers**:
This platform can be used to upload images of scenes, people, or items. Based on these images, the platform generates a short, creative story. 

### **Demo Walkthrough**:
1. **Uploading Images**: 
   - Select images and click on “Upload”.
2. **Story Generation**: 
   - After uploading, the system will generate a story based on the detected objects in the images.
3. **Live Demo**:
   - The live project can be accessed [here] https://drive.google.com/file/d/1fFIFbKk2oPQRGmMzyjDJs0NqqBZvxj8L/view?usp=drive_link.

---

## **g. Code Structure**

Here’s an overview of the project directory structure:

```
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── uploads/             # Folder to store uploaded images
├── templates/           # HTML templates for rendering the app
│   └── index.html
├── static/              # Static assets (CSS, JS)
└── README.md            # Documentation
```

### **Key Files**:
- `app.py`: The main application that handles image uploads, object detection, and story generation.
- `index.html`: The HTML page for uploading images and displaying results.

---

## **h. Testing**

### **Testing Process**:
- Manual testing of image uploads and story generation.
- Unit tests (if added) would cover the image processing and text generation steps.

### **Tools**:
- **Postman**: For testing API endpoints.
- **pytest**: For running unit tests.

---

## **i. Challenges and Solutions**

### **Challenges**:
- **Model Size and Inference Speed**: YOLOv8 and GPT-2 models are large and can take time for inference, especially on CPUs.
- **Handling Multiple Image Inputs**: Ensuring that the system can handle 3-10 images without crashing or slowing down.

### **Solutions**:
- Used GPU for faster processing.
- Optimized Flask app to handle multiple file uploads simultaneously.

---

## **j. Future Enhancements**

- **Enhanced Story Generation**: Integrate more complex models like GPT-3 for even more engaging and contextual story generation.
- **User Authentication**: Allow users to save and share stories.
- **Mobile App Integration**: Develop a mobile app for easy uploading and story sharing.

---

## **k. Credits and References**

- **Team Members**: Avvaru Dona Krishna Sai, Kuthyaru Lakshmi Janardhan, Guduru Mounika, Abdul Nafisa Sulthana
- **Libraries Used**:
  - YOLOv8 (Ultralytics)
  - HuggingFace Transformers (GPT-2)
  - Flask
  - OpenCV

---

This README should provide users and developers with the necessary information to understand, set up, and contribute to your project.
