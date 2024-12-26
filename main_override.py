import sys
import numpy as np
import pyodbc
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from model_util import DeepModel
import os

# Add your module path
sys.path.append('C:/Users/HP/Desktop/image-similarity/image-similarity')

# Initialize FastAPI app
app = FastAPI()

# Database Configuration
DATABASE_CONFIG = {
    "driver": "SQL Server",
    "server": "LAPTOP-U39SH27E\\NITIN",
    "database": "Products",
    "table_name": "XRayImages",
    "trusted_connection": "yes"
}

# Directory to store images
IMAGE_STORAGE_PATH = "C:/Users/HP/Desktop/XRayImages/"

# Ensure the directory exists
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)


# Function to connect to the database
def connect_to_database():
    try:
        connection = pyodbc.connect(
            f"Driver={{{DATABASE_CONFIG['driver']}}};"
            f"Server={DATABASE_CONFIG['server']};"
            f"Database={DATABASE_CONFIG['database']};"
            f"Trusted_Connection={DATABASE_CONFIG['trusted_connection']};"
        )
        return connection
    except Exception as e:
        raise Exception(f"Database connection error: {e}")


# Function to fetch file paths for a product
def fetch_file_paths(product_code):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Query to fetch file paths for the given product code
        query = f"SELECT filePath FROM {DATABASE_CONFIG['table_name']} WHERE productCode = ?"
        cursor.execute(query, product_code)

        # Fetch all file paths
        file_paths = [row.filePath for row in cursor.fetchall()]
        return file_paths

    except Exception as e:
        print(f"Database error: {e}")
        return None


# Function to save file path and product code to the database
def save_to_database(product_code, file_path):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Insert query
        query = f"INSERT INTO {DATABASE_CONFIG['table_name']} (productCode, filePath) VALUES (?, ?)"
        cursor.execute(query, (product_code, file_path))
        connection.commit()

    except Exception as e:
        print(f"Database error: {e}")


# Image similarity processing class
class ImageSimilarityBatch:
    @staticmethod
    def process_images_batch(target_image_path, file_paths):
        """
        Processes a target image and compares it to an array of other images, calculating similarity scores.
        """
        try:
            # Initialize the model
            model = DeepModel()

            # Preprocess the target image
            target_feature = DeepModel.preprocess_image(target_image_path)
            target_feature = np.expand_dims(target_feature, axis=0)  # Add batch dimension
            target_feature = model.extract_feature([target_feature])[0]  # Extract feature

            # Iterate through the list of file paths
            for file_path in file_paths:
                # Preprocess each image
                feature = DeepModel.preprocess_image(file_path)
                feature = np.expand_dims(feature, axis=0)  # Add batch dimension
                feature = model.extract_feature([feature])[0]  # Extract feature

                # Calculate similarity
                similarity = ImageSimilarityBatch.calculate_similarity(target_feature, feature)

                # If similarity is above the threshold, return "Matched"
                if similarity > 0.75:
                    return "Matched"

            # If no matches found, return "Did not match"
            return "Did not match"

        except Exception as e:
            print(f"Error processing images: {e}")
            return None

    @staticmethod
    def calculate_similarity(feature1, feature2):
        """
        Calculates the cosine similarity between two feature vectors.
        """
        from numpy.linalg import norm
        from numpy import dot

        if feature1 is None or feature2 is None:
            raise ValueError("Features cannot be None")

        return dot(feature1, feature2) / (norm(feature1) * norm(feature2))


# FastAPI Endpoints

@app.post("/check_similarity/")
async def check_similarity(product_code: str = Form(...), target_image: UploadFile = File(...)):
    """
    Endpoint to check similarity for the given product_code and target image.
    """
    try:
        # Save the uploaded target image temporarily
        target_image_path = f"{IMAGE_STORAGE_PATH}{target_image.filename}"
        with open(target_image_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)

        # Fetch file paths for the product code
        file_paths = fetch_file_paths(product_code)

        if file_paths:
            # Process the images for similarity
            result = ImageSimilarityBatch.process_images_batch(target_image_path, file_paths)
            return JSONResponse({"status": result})
        else:
            return JSONResponse({"status": "No file paths found for the given product code"})

    except Exception as e:
        return JSONResponse({"status": "Error", "details": str(e)})



@app.post("/add_image/")
async def add_image(product_code: str = Form(...), image_file: UploadFile = File(...)):
    """
    Endpoint to add a new image for a given product_code.
    """
    try:
        # Save the uploaded image
        saved_file_path = f"{IMAGE_STORAGE_PATH}{image_file.filename}"
        with open(saved_file_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Save the file path and product code to the database
        save_to_database(product_code, saved_file_path)

        return JSONResponse({"status": "Image added successfully"})

    except Exception as e:
        return JSONResponse({"status": "Error", "details": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
