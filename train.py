import face_recognition
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the path to the uploads folder
uploads_folder = 'uploads1'
reference_encodings = {}
students = {}

# Step 1: Load the training images and extract encodings
for image_file in os.listdir(uploads_folder):
    student_id = os.path.splitext(image_file)[0]  # Use the file name (without extension) as the student ID (roll number)
    image_path = os.path.join(uploads_folder, image_file)
    
    # Load the image and extract face encodings
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        reference_encodings[student_id] = face_encodings[0]
        students[student_id] = student_id  # Store student information
    else:
        print(f"No face found in image for student {student_id} at {image_path}")

# Step 2: Test the model using the same images
test_labels = []
predicted_labels = []

def test_accuracy():
    for image_file in os.listdir(uploads_folder):
        student_id = os.path.splitext(image_file)[0]  # Use the file name (without extension) as the student ID (roll number)
        image_path = os.path.join(uploads_folder, image_file)
        
        # Load the test image
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            face_encoding = face_encodings[0]
            # Compare the face encoding with the reference encodings
            matches = face_recognition.compare_faces(list(reference_encodings.values()), face_encoding)
            face_distances = face_recognition.face_distance(list(reference_encodings.values()), face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                predicted_student_id = list(reference_encodings.keys())[best_match_index]
            else:
                predicted_student_id = "Unknown"
        else:
            predicted_student_id = "No Face Detected"

        # Log image processing status
        print(f"Image: {image_file}, Detected Faces: {len(face_encodings)}, Predicted: {predicted_student_id}")

        # Append the actual and predicted labels
        test_labels.append(student_id)  # The correct student ID (roll number)
        predicted_labels.append(predicted_student_id)

test_accuracy()

# Step 3: Check the accuracy
filtered_test_labels = []
filtered_predicted_labels = []

for true_label, predicted_label in zip(test_labels, predicted_labels):
    if predicted_label not in ["Unknown", "No Face Detected"]:
        filtered_test_labels.append(true_label)
        filtered_predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = accuracy_score(filtered_test_labels, filtered_predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(filtered_test_labels, filtered_predicted_labels))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(filtered_test_labels, filtered_predicted_labels))
