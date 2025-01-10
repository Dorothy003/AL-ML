import os
import cv2
import face_recognition
import numpy as np
from imgaug import augmenters as iaa
import bson
from datetime import datetime
from pymongo import MongoClient

# MongoDB Setup
MONGO_URI = "mongodb://127.0.0.1:27017/attendance_system"
client = MongoClient(MONGO_URI)
db = client['attendance_system']
students_collection = db['students']

# Path where student photos will be saved
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads1')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Augmentation function for images (optional, helps with model generalization)
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-30, 30)),  # rotate images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # add gaussian noise
    ])
    return seq.augment_image(image)

# Function to process images and extract face encodings
def preprocess_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings

# Save student to MongoDB
def save_student(student_id, student_data):
    students_collection.update_one(
        {'_id': student_id},
        {'$set': student_data},
        upsert=True
    )

# Train the system with students' images
def train_system():
    # Dictionary to hold the face encodings of all students
    reference_encodings = {}

    # Iterate through all student images and add them to the reference encoding database
    for student in students_collection.find():
        student_id = str(student['_id'])
        student_name = student['name']
        image_filename = student['image']
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        # Preprocess the image and extract face encodings
        encodings = preprocess_image(image_path)
        if encodings:
            reference_encodings[student_id] = encodings[0]
            print(f"Encoding added for {student_name}")
        else:
            print(f"No face found for {student_name}")

    return reference_encodings

# Train and save the encodings
def save_face_encodings_to_file(reference_encodings):
    with open('face_encodings.npy', 'wb') as f:
        np.save(f, reference_encodings)
    print("Face encodings saved to file.")

# Load encodings from file (for future use)
def load_face_encodings_from_file():
    try:
        with open('face_encodings.npy', 'rb') as f:
            return np.load(f, allow_pickle=True).item()
    except FileNotFoundError:
        print("Face encodings file not found.")
        return {}

# Test the system with a list of images (for testing accuracy)
def test_system(reference_encodings, test_images, true_labels):
    predictions = []
    for image_path in test_images:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if face_encodings:
            # Find the closest match to the reference encodings
            distances = face_recognition.face_distance(list(reference_encodings.values()), face_encodings[0])
            best_match_index = np.argmin(distances)
            student_id = list(reference_encodings.keys())[best_match_index]
            predictions.append(student_id)
    
    # Calculate and print accuracy
    accuracy = calculate_accuracy(predictions, true_labels)
    print(f"Accuracy: {accuracy}%")

def calculate_accuracy(predictions, true_labels):
    """
    Calculate the accuracy of predictions compared to true labels.
    
    Args:
    predictions (list): List of predicted student IDs or names.
    true_labels (list): List of true student IDs or names.
    
    Returns:
    float: The accuracy as a percentage.
    """
    correct_predictions = 0
    total_predictions = len(predictions)
    
    # Ensure the length of predictions matches true_labels
    if total_predictions != len(true_labels):
        print("Error: The number of predictions does not match the number of true labels.")
        return None
    
    # Count the number of correct predictions
    for predicted, true in zip(predictions, true_labels):
        if predicted == true:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Main function to run the training and testing process
def main():
    # First, make sure the student data is saved (this is where students are registered manually or by uploading photos)
    students = [
        # Add some sample students
    ]

    # Insert students into MongoDB
    for student in students:
        student_data = {
            'name': student['name'],
            'student_id': student['student_id'],
            'image': student['image']
        }
        student_id = bson.ObjectId()  # Unique student ID for MongoDB
        student_data['_id'] = student_id
        save_student(student_id, student_data)

    # After registering students, train the system
    reference_encodings = train_system()

    # Save encodings to file for later use
    save_face_encodings_to_file(reference_encodings)
    
    # Now, test the system and calculate accuracy
    test_images = [
        # Paths to test images
    ]
    
    true_labels = [
        # Corresponding true labels (student IDs)
    ]
    
    test_system(reference_encodings, test_images, true_labels)
    
if __name__ == '__main__':
    main()
