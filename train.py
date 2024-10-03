from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import numpy as np
import os
import json
import face_recognition
from datetime import datetime
from imgaug import augmenters as iaa
def calculate_accuracy(test_images):
    correct_predictions = 0
    total_images = len(test_images)

    for student_id, test_image_path in test_images.items():
        # Load the test image
        test_image = face_recognition.load_image_file(test_image_path)
        test_encoding = face_recognition.face_encodings(test_image)
        
        if not test_encoding:
            print(f"No face found in test image for student ID: {student_id}.")
            continue

        test_encoding = test_encoding[0]

        # Compare the test image encoding with the reference encodings
        matches = face_recognition.compare_faces(list(reference_encodings.values()), test_encoding)
        if True in matches:
            matched_index = matches.index(True)
            predicted_student_id = list(reference_encodings.keys())[matched_index]
            if predicted_student_id == student_id:
                correct_predictions += 1

    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    return accuracy
