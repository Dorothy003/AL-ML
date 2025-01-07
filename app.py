from pymongo import MongoClient
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
MONGO_URI = "mongodb://127.0.0.1:27017/attendance_system"
try:
 client = MongoClient(MONGO_URI)
 db = client['attendance_system']
 students_collection = db['students']
 attendance_collection = db['attendance']
 attendance_log_collection = db['attendance_log']
 print("MongoDB connected succesfully")
except Exception as e:
 print(f"Error connecting to mongodb: {e}")
 raise
# Ensure the upload folder exists
upload_folder_path = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder_path):
    os.makedirs(upload_folder_path)

# Load student data
def load_students():
  
         students = {}
         for student in students_collection.find():
             students[str(student['_id'])] = {
                 'name': student['name'],
                 'image': student['image']
             }
         return students

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


#students = load_students()
def save_students(student_id, student_data):
    students_collection.update_one(
        {'_id': student_id},
        {'$set': student_data},
            
        upsert = True
    )


def load_attendance_log():
    # Assuming you want to load the attendance log from MongoDB
    attendance_log_collection = db.attendance_log_collection
    
    # Fetch the entire attendance log
    attendance_log = {}
    
    # Use a query to find all documents
    for record in attendance_log_collection.find():
        date = record.get('date')
        attendance_log[date] = record.get('daily_log', {})
    
    return attendance_log

                                            


attendance_log = load_attendance_log()
def save_attendance_log(date, student_name, status):
    attendance_collection.update_one(
        {'date': date, 'student_name': student_name},
        {'$set':{'status': status}},
        upsert=True
            
        
    )
students = load_students()
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

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('mark_as_present')
def handle_mark_present(data):
    student_name = data.get('name')
    if student_name:
        date = datetime.now().strftime("%Y-%m-%d")
   
        save_attendance_log(date, student_name, 'Present')
        print(f"{student_name} marked as present")

@socketio.on('mark_as_absent')
def handle_mark_absent(data):
    student_name = data.get('name')
    if student_name:
        date = datetime.now().strftime("%Y-%m-%d")
        #attendance_log.setdefault(date, {})[student_name] = 'Absent'
        #save_attendance_log(attendance_log)
        save_attendance_log(date, student_name, 'Absent')
        print(f"{student_name} marked as absent")

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
                
                # Emit recognized student name to the frontend for confirmation
                socketio.emit('recognized_student', {'name': recognized_student})

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
                #save_students(students)
                save_students(student_id,{'name': student_name, 'image': filename})
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

    # Update attendance log based on status
    if status == 'Present':
        attendance_log[date][student_name] = 'Present'
    elif status == 'Absent':
        attendance_log[date].pop(student_name, None)  # Remove if exists

    # Save the updated attendance log
    save_attendance_log(attendance_log)

    return jsonify({'message': f'Attendance for {student_name} on {date} marked as {status}'}), 200



@app.route('/video_feed_page')
def video_feed_page():
    return render_template('video_feed.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def load_json_data():
    with open('students1.json', 'r') as students_file, open('attendance_log.json', 'r') as attendance_file:
        students = json.load(students_file)
        attendance_log = json.load(attendance_file)
    return students, attendance_log
@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    students, attendance_log = load_json_data()
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

def clean_attendance_log():
  
    for record in attendance_log_collection.find():  # Correct MongoDB collection query
        date = record.get('date')
        daily_log = record.get('daily_log', None)

        if not daily_log:  # If daily_log is None or empty, skip the record
            print(f"Invalid or missing 'log' for record with date {date}. Skipping...")
            continue
        
        # Loop through each student and mark attendance
        for student_name in daily_log:
            # Use update_one to update attendance status for each student
            attendance_log_collection.update_one(
                {"date": date},  # Filter by date
                {"$set": {f"attendance.{student_name}": "Present"}},  # Update attendance for the student
                upsert=True 
            )
    
    print("Attendance log cleaned successfully.")


if __name__ == "__main__":
    clean_attendance_log()
    socketio.run(app, debug=True)
