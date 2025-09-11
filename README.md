
# 🌱 Plant Verification System – README

## 📖 Overview
This project implements an AI-powered image verification system to confirm that a tree or plant has been planted in a specified area. Users upload two images:

1. **Before Image** – of the location before planting.
2. **After Image** – of the same location after planting.

The system uses AI models and image processing techniques to verify that:

✔ The two images are of the same area  
✔ The area was free of plants in the first image  
✔ A plant/tree is present in the second image  
✔ The plant/tree is located in the same area as in the first image

## 📂 How It Works

The process follows the flowchart below:

1. **Capture Before Image**  
   The user uploads an image of the area before planting.

2. **Plant Tree**  
   The user plants a tree or plant in the same location.

3. **Capture After Image**  
   The user uploads a second image of the same location after planting.

4. **Check Area Similarity**  
   Uses deep learning models such as VGG16, MobileNet, or DenseNet121 to compare the images.  
   If no match → Reject request.  
   If match → Proceed to next steps.  
   If unable to process → Handle accordingly.

5. **Check if Plant is Absent in Image 1**  
   Uses object detection models like Hugging Face or TensorFlow Hub to ensure no plant/tree is present.  
   If plant/tree is present → Flag for fraud.  
   If absent → Proceed.

6. **Check if Plant is Present in Image 2**  
   Uses object detection models to confirm the plant/tree is visible.  
   If absent → Reject request.  
   If present → Proceed.

7. **Check if Plant Location Overlaps Area**  
   Uses OpenCV’s feature matching algorithms like ORB, SIFT, or AKAZE.  
   If overlap → Verification passed!  
   If not overlap → Flag for review.

## ✅ Technologies Used

| Task                        | Technology / Library      |
|----------------------------|---------------------------|
| Area similarity            | VGG16, MobileNet, DenseNet121 |
| Object detection           | Hugging Face API, TensorFlow Hub |
| Feature matching           | OpenCV (ORB, SIFT, AKAZE) |
| Review and fraud handling  | Manual review or alerts  |

## 📦 Files

- **AI verification flowchart image** – Visual representation of the workflow.
- **Backend service code** – API endpoints for image upload and processing.
- **Frontend interface** – Web application for user interaction.

## 🚀 How to Run

1. Clone the repository.
2. Set up the environment with required dependencies (`tensorflow`, `opencv-python`, `requests`, etc.).
3. Set the API keys (e.g., Hugging Face) in environment variables.
4. Run the backend service.
5. Access the frontend and upload images for verification.

## 📥 Installation

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
export HF_API_TOKEN="your_huggingface_api_token_here"
python main_override.py
```

## ⚙ Configuration

Add your Hugging Face API token to an environment file:

```bash
HF_API_TOKEN=your_actual_token
```

Or export it directly:

```bash
export HF_API_TOKEN="your_actual_token"
```

## ✅ Key Features

✔ Image-based verification using AI models  
✔ Fraud detection by analyzing image contents  
✔ Area similarity check using deep learning features  
✔ Spatial alignment verification using OpenCV  
✔ Extensible architecture with API integration  

## 📂 Future Improvements

✔ Add support for image metadata (GPS, timestamp)  
✔ Improve object detection accuracy with custom models  
✔ Integrate multi-factor authentication for users  
✔ Enhance fraud prevention through pattern analysis  
✔ Deploy the solution using Docker or cloud platforms  

## 📞 Contact

For any questions or contributions, contact:

**Nitin Tripathi**  
Email: nitintripathi2710@gmail.com
