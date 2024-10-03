from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import json
import face_recognition
from datetime import datetime
from imgaug import augmenters as iaa

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads1')  # Used os.getcwd() to get the current working directory
app.secret_key = 'supersecretkey'  # Needed for flash messages
socketio = SocketIO(app)  # Initialize Flask-SocketIO

BASE_DIR = os.getcwd()

# Ensure the upload folder exists
upload_folder_path = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder_path):
    os.makedirs(upload_folder_path)

# Load student data
def load_students():
    students_file_path = os.path.join(BASE_DIR, 'students1.json')
    if not os.path.exists(students_file_path):
        return {}
    with open(students_file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            return {}

def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-30, 30)),  # rotate images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # add gaussian noise
    ])
    return seq.augment_image(image)
def preprocess_image(image_path):
    image = face_recognition.load_image_file(image_path)
    # Resize image or perform other preprocessing...
    return image
def save_students(students):
    students_file_path = os.path.join(BASE_DIR, 'students1.json')
    try:
        with open(students_file_path, 'w') as f:
            json.dump(students, f, indent=4)  # Added indent for better readability
    except Exception as e:
        print(f"Error saving students file: {e}")
        flash(f'Error saving student data: {e}', 'error')

students = load_students()

# Load attendance log
def load_attendance_log():
    attendance_log_path = os.path.join(BASE_DIR, 'attendance_log.json')
    if not os.path.exists(attendance_log_path):
        return {}
    with open(attendance_log_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading attendance log: {e}")
            return {}

def save_attendance_log(attendance_log):
    attendance_log_path = os.path.join(BASE_DIR, 'attendance_log.json')
    try:
        with open(attendance_log_path, 'w') as f:
            json.dump(attendance_log, f, indent=4)
    except Exception as e:
        print(f"Error saving attendance log file: {e}")

attendance_log = load_attendance_log()

# Load reference encodings
reference_encodings = {}
for student_id, student_data in students.items():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], student_data['image'])
    if os.path.exists(image_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:  # Check if encodings is not empty
            reference_encodings[student_id] = encodings[0]
        else:
            print(f"No face found in image for student {student_data['name']} at {image_path}")
    else:
        print(f"Image for student {student_data['name']} not found at {image_path}")
        

recent_recognitions = {}

def generate_frames():
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not video_capture.isOpened():
        print("Error: Could not open video capture device.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_student = None
        for face_encoding in face_encodings:
            # Compare the face encoding with the reference encodings
            matches = face_recognition.compare_faces(list(reference_encodings.values()), face_encoding)
            face_distances = face_recognition.face_distance(list(reference_encodings.values()), face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                student_id = list(reference_encodings.keys())[best_match_index]
                recognized_student = students[student_id]['name']
                mark_attendance(recognized_student)
                break

        # Display the result
        if recognized_student:
            cv2.putText(frame, f"Match: {recognized_student}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()



def mark_attendance(student_name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    if date_str not in attendance_log:
        attendance_log[date_str] = {}
    # Check if the student is already present today
    if student_name not in attendance_log[date_str]:
        attendance_log[date_str][student_name] = time_str
        save_attendance_log(attendance_log)

@app.route('/')
@app.route('/Home')
def home():
    return render_template('home.html')

@app.route('/Addstudent', methods=['GET', 'POST'])
def Addstudent():
    if request.method == 'POST':
        student_id = request.form['studentid']
        student_name = request.form['studentname']

        # Ensure the student ID and name are provided
        if not student_id or not student_name:
            flash('Student ID and Name are required', 'error')
            return redirect(request.url)

        if 'studentphoto' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['studentphoto']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            try:
                filename = f"{student_id}.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Confirm file is saved
                if not os.path.exists(file_path):
                    flash('File was not saved correctly', 'error')
                    return redirect(request.url)

                # Save student data
                students[student_id] = {'name': student_name, 'image': filename}
                save_students(students)
                
                # Update reference encodings
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)[0]
                reference_encodings[student_id] = encoding

                flash('Student added successfully!', 'success')
                return redirect(url_for('home'))  # Ensure to redirect after successful add
            except Exception as e:
                flash(f'Error adding student: {e}', 'error')
                print(f"Error: {e}")

    # If GET request or no action taken, return the Addstudent page
    return render_template('Addstudent.html')

@app.route('/Addsubject')
def Addsubject():
    return render_template('Addsubject.html')

@app.route('/Attendancelog')
def Attendancelog():
    dates = sorted(attendance_log.keys())
    return render_template('Attendancelog.html', students=students, attendance_log=attendance_log, dates=dates)

@app.route('/mark_attendance_ajax', methods=['POST'])
def mark_attendance_ajax():
    data = request.get_json()
    student_id = data['student_id']
    date = data['date']
    status = data['status']

    if date not in attendance_log:
        attendance_log[date] = {}
    
    student_name = students[student_id]['name']
    if status == 'Present':
        attendance_log[date][student_name] = 'Present'
    else:
        if student_name in attendance_log[date]:
            del attendance_log[date][student_name]

    # Save attendance log
    save_attendance_log(attendance_log)

    return jsonify({'message': f'Attendance for {student_name} on {date} marked as {status}'}), 200

@app.route('/subject_video_feed/<subject>')
def subject_video_feed(subject):
    return Response(generate_frames_for_subject(subject), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_page')
def video_feed_page():
    return render_template('video_feed.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    selected_date = None
    attendance_status = {}
    total_attendance_count = {student_data['name']: 0 for student_data in students.values()}

    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
        if selected_date:
            if selected_date in attendance_log:
                attendance_status = attendance_log[selected_date]
            else:
                flash(f'No attendance record found for {selected_date}.', 'error')

    # Calculate total attendance
    for date, daily_log in attendance_log.items():
        for student_name in daily_log.keys():
            total_attendance_count[student_name] += 1

    # Include today's date
    today_date = datetime.now().strftime("%Y-%m-%d")
    dates = sorted(list(attendance_log.keys()) + [today_date])  # Convert dict_keys to list and include today’s date

    return render_template('view_attendance.html', dates=dates, 
                           attendance_status=attendance_status, selected_date=selected_date, 
                           total_attendance_count=total_attendance_count, students=students)

if __name__ == "__main__":
    socketio.run(app, debug=True)
