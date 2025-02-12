from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify,session
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import json
import face_recognition
from datetime import datetime
from imgaug import augmenters as iaa
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import sys
sys.dont_write_bytecode = True
from pymongo import MongoClient
import uuid

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Log-in"
app.secret_key = os.urandom(24)
mongo = PyMongo(app)



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

def load_students():
    students = {}
    try:
        for student in students_collection.find():
            student_id = str(student.get('_id', 'Unknown_ID'))  # Use 'Unknown_ID' if '_id' is missing.
            
            # Validate and log missing fields for better debugging.
            if not student_id or student_id == 'Unknown_ID':
                print(f"Missing student ID for document: {student}")
            
            students[student_id] = {
                'subject_name': student.get('subject_name', 'Unknown'),
                'name': student.get('name', 'Unknown'),
                'image': student.get('image', 'default.jpg'),
                'subject_code': student.get('subject_code', 'N/A')
            }
    except Exception as e:
        print(f"Error loading students: {e}")
    
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
def save_students(student_id, student_data):
    students_collection.update_one(
        {'_id': student_id},
        {'$set': student_data},
            
        upsert = True
    )

# Load attendance log
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
def save_attendance_log(date, student_name, status, subject_name):
    # Insert or update attendance collection with the new attendance data
    attendance_collection.update_one(
        {'date': date, 'student_name': student_name, 'subject_name': subject_name},
        {'$set': {'status': status}},
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

from datetime import datetime
@socketio.on('mark_as_present')
def handle_mark_present(data):
    student_name = data.get('name')
    subject_name = data.get('subject_name')  # Get the selected subject name

    if student_name and subject_name:
        # Fetch the student record for the given name and subject
        student = students_collection.find_one({
            'name': student_name,
            'subject_name': subject_name
        })
        
        if student:
            # Student found and enrolled in the given subject
            date = datetime.now().strftime("%Y-%m-%d")
            
            # Save the attendance log to the database for the selected subject
            save_attendance_log(date, student_name, 'Present', subject_name)
            
            # Log the action in the server
            print(f"{student_name} marked as present for subject {subject_name} on {date}")
            
            # Send success response back to the client
            socketio.emit('attendance_marked', {
                'status': 'success',
                'message': f'Attendance marked for {student_name} on {date}'
            })
        else:
            # Handle case where the student is not enrolled in the selected subject
            print(f"Error: {student_name} is not enrolled in {subject_name}")
            socketio.emit('attendance_marked', {
                'status': 'error',
                'message': f'{student_name} is not enrolled in {subject_name}'
            })
    else:
        # Handle missing student name or subject name
        print(f"Error: Missing student name or subject name in data: {data}")
        socketio.emit('attendance_marked', {
            'status': 'error',
            'message': 'Missing student name or subject name'
        })

# Mark student as absent
@socketio.on('mark_as_absent')
def handle_mark_absent(data):
    student_name = data.get('name')
    subject_name = data.get('subject_name')  # Get the subject name

    if student_name and subject_name:
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if the student is enrolled in the subject
        student = students_collection.find_one({'name': student_name})

        if student:
            # Check if the student is enrolled in the selected subject
            if subject_name in [subject['subject_name'] for subject in student.get('subjects', [])]:
                # Handle absence for the student and subject
                attendance_collection.update_one(
                    {'date': date, 'student_name': student_name, 'subject_name': subject_name},
                    {'$set': {'status': 'Absent'}},
                    upsert=True
                )
                print(f"{student_name} marked as absent for {subject_name} on {date}")
                socketio.emit('attendance_marked', {'status': 'success', 'message': f'{student_name} marked as absent for {subject_name}'})
            else:
                # Student is not enrolled in the selected subject
                socketio.emit('attendance_marked', {'status': 'error', 'message': f'{student_name} is not enrolled in {subject_name}'})
        else:
            # Handle case where student is not found in the database
            socketio.emit('attendance_marked', {'status': 'error', 'message': f'{student_name} not found'})
    else:
        # Handle missing student name or subject name
        socketio.emit('attendance_marked', {'status': 'error', 'message': 'Missing student name or subject name'})

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
def index():
    if 'user' not in session:
        return redirect(url_for('login')) 
    return redirect(url_for('dashboard')) 
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    email = session['user']

    # Fetch user data from MongoDB
    user = mongo.db.users.find_one({'email': email})

    if user:
        homepage_id = user.get('homepage_id')  # Get homepage_id from MongoDB document
        return render_template('home.html', user=user, homepage_id=homepage_id)
    else:
        return redirect(url_for('login'))

@app.route('/Addstudent', methods=['GET', 'POST'])
def Addstudent():
    if 'user' not in session:
        flash('Please log in to add students.', 'error')
        return redirect(url_for('login'))
    
    user_email = session['user']
    user = mongo.db.users.find_one({'email': user_email})
    homepage_id = user.get('homepage_id')
    if request.method == 'POST':
        try:
            student_id = request.form['studentid']
            student_name = request.form['studentname']
            subject_code = request.form['subject_code']

            # Validate file input
            if 'studentphoto' not in request.files or request.files['studentphoto'].filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)

            file = request.files['studentphoto']
            filename = f"{student_id}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Confirm file saved
            if not os.path.exists(file_path):
                flash('File was not saved correctly.', 'error')
                return redirect(request.url)

            # Find the subject name based on subject_code
            subject_name = None
            for semester in user.get('semesters', []):
                for subject in semester.get('subjects', []):
                    if subject['subject_code'] == subject_code:
                        subject_name = subject['subject_name']
                        break
                if subject_name:
                    break

            if not subject_name:
                flash('Subject not found.', 'error')
                return redirect(request.url)

            # Create student entry
            student_entry = {
                "id": student_id,
                "image": filename,
                "name": student_name,
                "subject_code": subject_code,
                "subject_name": subject_name,
                "user_id": homepage_id
            }

            # Ensure no auto-generated `_id` field by using `replace_one`
            students_collection.replace_one(
                {"id": student_id, "subject_code": subject_code},  # Match on student_id and subject_code
                student_entry,
                upsert=True  # Insert if no match is found
            )

            flash('Student added successfully!', 'success')
            return redirect(url_for('home'))

        except Exception as e:
            flash(f'Error adding student: {e}', 'error')
            print(f"Error: {e}")
            return redirect(request.url)

    return render_template('Addstudent.html', semesters=user.get('semesters', []))


import uuid

import uuid
from flask import session, redirect, url_for, flash, request

import uuid
from flask import session, redirect, url_for, flash, request

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if the user exists in the database
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            # Check if homepage_id already exists
            if 'homepage_id' not in user:
                # If homepage_id does not exist, generate and save it
                homepage_id = str(uuid.uuid4())  # Generates a unique ID
                mongo.db.users.update_one(
                    {'email': email},
                    {'$set': {'homepage_id': homepage_id}}
                )
            else:
                # If homepage_id exists, use the existing one
                homepage_id = user['homepage_id']
            
            # Store the user's email and homepage_id in the session
            session['user'] = email
            session['homepage_id'] = homepage_id  # Store homepage_id in session

            return redirect(url_for('home'))
        else:
            flash('Invalid Email or Password. Please try again.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']
        password_hash = generate_password_hash(password)  # Store hashed password
        
        # Check if user already exists
        if mongo.db.users.find_one({'email': email}):
            flash('Email already exists. Please choose a different email.', 'error')
            return redirect(url_for('register'))
        
        # Insert new user into the database
        user_id = mongo.db.users.insert_one({'name': name, 'email': email, 'password': password_hash}).inserted_id

        
        semesters = ['Semester 1', 'Semester 2', 'Semester 3', 'Semester 4', 'Semester 5', 'Semester 6', 'Semester 7', 'Semester 8']
        for semester_name in semesters:
            mongo.db.users.update_one(
                {'_id': user_id},
                {'$push': {'semesters': {'name': semester_name, 'subjects': []}}}
            )
        
        flash('Registration Successful!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
   
    return redirect(url_for('login'))

@app.route('/Addsubject', methods=['GET', 'POST'])
def Addsubject():
    if request.method == 'POST':
        subject_name = request.form.get('subjectname')
        semester = request.form.get('semester')

        # Validate the input
        if not subject_name or not semester:
            flash('Subject name and semester are required.', 'error')
            return redirect(url_for('Addsubject'))

        # Save to MongoDB
        subject_data = {'name': subject_name, 'semester': semester}
        db.subjects.insert_one(subject_data)

        flash('Subject added successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('Addsubject.html')
@app.route('/Attendancelog', methods=['GET', 'POST'])
def Attendancelog():
    # Get the logged-in user's homepage_id
    homepage_id = session.get('homepage_id')  # Assuming homepage_id is stored in the session
    
    # Fetch the user's data from the MongoDB log_in collection using homepage_id
    user_data = mongo.db.users.find_one({"homepage_id": homepage_id})

    # Initialize subjects and students_by_subject
    subjects = []
    students_by_subject = {}

    if user_data:
        # Loop through the user's semesters and subjects
        for semester in user_data.get('semesters', []):
            for subject in semester.get('subjects', []):
                subjects.append(subject)  # Collect subjects
                
                # Fetch students enrolled in this subject from the students collection
               # students = students_collection.find({'subject_name': subject['subject_name']})
                students = students_collection.find({
                    'subject_name': subject['subject_name'],
                    'user_id': homepage_id
                })
                students_by_subject[subject['subject_name']] = [
                    {
                        'student_id': student.get('id'),
                        'name': student.get('name'),
                        'image': student.get('image')
                    }
                    for student in students
                ]

    # Handle subject filter (if any)
    subject_filter = request.args.get('subject_name')
    if subject_filter:
        # Filter the students and attendance for the selected subject
        students_by_subject = {subject: students for subject, students in students_by_subject.items() if subject == subject_filter}

    # Fetch attendance records from the attendance collection
    attendance_records = list(db.attendance.find({}))
    
    # Initialize the attendance log to store attendance by student_id for each date
    attendance_log = {}

    for record in attendance_records:
        student_name = record.get('student_name')
        subject_name = record.get('subject_name')  # Include subject name in validation
        date = record.get('date')
        status = record.get('status', "Absent")  # Default to "Absent" if status is missing
        
        if not student_name or not date or not subject_name:
            continue  # Skip if student_name, date, or subject_name is missing
        
        # Initialize attendance log for each date
        if date not in attendance_log:
            attendance_log[date] = {}

        # Find student_id based on student_name and subject_name
        student_id = None
        for subject, students in students_by_subject.items():
            if subject == subject_name:  # Ensure the attendance is tied to the correct subject
                for student_data in students:
                    if student_data['name'] == student_name:
                        student_id = student_data['student_id']
                        break
        
        # Log attendance for the student in the attendance log
        if student_id:
            # We store the attendance status under the student's id for that subject
            if student_id not in attendance_log[date]:
                attendance_log[date][student_id] = {}  # Initialize student attendance record for that date
            attendance_log[date][student_id][subject_name] = status  # Log the attendance status

    # Sort the dates for the final display
    dates = sorted(attendance_log.keys())

    # Pass the subjects, students_by_subject, and filtered data to the template
    return render_template(
        'Attendancelog.html',
        attendance_log=attendance_log,
        dates=dates,
        subjects=subjects,
        students_by_subject=students_by_subject
    ) 


@app.route('/mark_attendance_ajax', methods=['POST'])
def mark_attendance_ajax():
    data = request.get_json()
    student_id = data.get('student_id')
    date = data.get('date')
    status = data.get('status')
    
    # Validate required fields
    if not student_id or not date or not status:
        return jsonify({'error': 'Invalid data received'}), 400

    # Fetch student details from the database
    student = students_collection.find_one({'_id': student_id})
    if not student:
        return jsonify({'error': 'Student not found'}), 404

    student_name = student.get('name', 'Unknown')
    subject_code = data.get('subject_code') or student.get('subject_code', 'N/A')
    subject_name = data.get('subject_name') or student.get('subject_name', 'Unknown Subject')

    # Create the attendance record
    attendance_record = {
        'student_id': student_id,
        'student_name': student_name,
        'date': date,
        'status': status,
        'subject_code': subject_code,
        'subject_name': subject_name,
        'timestamp': datetime.now()  # Store as a proper datetime object
    }

    # Insert attendance record into the database
    attendance_collection.insert_one(attendance_record)
    print(f"Attendance recorded in MongoDB: {attendance_record}")

    return jsonify({'message': f'Attendance for {student_name} on {date} marked as {status}'}), 200


@app.route('/video_feed_page', methods=['GET', 'POST'])
def video_feed_page():
    if 'user' not in session:
        flash('Please log in to view video feed.', 'error')
        return redirect(url_for('login'))

    user_email = session['user']
    user = mongo.db.users.find_one({'email': user_email})

    if request.method == 'POST':
        selected_subject_code = request.form['subject_code']
        # You can handle any processing for the selected subject here

    return render_template('video_feed.html', semesters=user.get('semesters', []))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def load_json_data():
    with open('students1.json', 'r') as students_file, open('attendance_log.json', 'r') as attendance_file:
        students = json.load(students_file)
        attendance_log = json.load(attendance_file)
    return students, attendance_log

@app.route('/delete_subject', methods=['POST'])
def delete_subject():
    semester_name = request.form['semester_name']
    subject_name = request.form['subject_name']

    # Query the user document to find the semester
    user_email = session.get('user')  # Get the logged-in user's email from the session
    user = mongo.db.users.find_one({'email': user_email})
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))
    
    # Find the semester document
    semester = next((sem for sem in user['semesters'] if sem['name'] == semester_name), None)
    
    if semester:
        # Find the subject within the semester and remove it
        subject = next((sub for sub in semester['subjects'] if sub.get('subject_name') == subject_name), None)
        if subject:
            semester['subjects'].remove(subject)
            mongo.db.users.update_one(
                {'email': user_email},
                {'$set': {'semesters': user['semesters']}}  # Update the semesters with the removed subject
            )
            flash('Subject deleted successfully!', 'success')
        else:
            flash('Subject not found.', 'error')
    else:
        flash('Semester not found.', 'error')

    return redirect(url_for('dashboard'))

@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    # Fetch all students and attendance records from MongoDB
    #homepage_id = session.get('homepage_id')
    students_records = list(db.students.find({}))
    attendance_records = list(db.attendance.find({}))
    
    # Create a dictionary to map student names and their IDs, along with enrolled subjects
    students = {}
    for record in students_records:
        student_id = str(record['_id'])
        student_name = record.get('name', "Unknown")
        subject_names = record.get('subjects', [])  # Assuming each student has a list of enrolled subjects
        students[student_id] = {'name': student_name, 'subjects': subject_names}
    
    # Initialize attendance_log and subjects_set
    attendance_log = {}
    subjects_set = set()
    for record in attendance_records:
        student_name = record.get('student_name')
        date = record.get('date')
        subject_name = record.get('subject_name')
        status = record.get('status', "Absent")
        
        if not student_name or not date or not subject_name:
            continue
        
        # Add subject to the subjects set
        subjects_set.add(subject_name)

        if date not in attendance_log:
            attendance_log[date] = {}

        if subject_name not in attendance_log[date]:
            attendance_log[date][subject_name] = {}

        attendance_log[date][subject_name][student_name] = status

    # Convert the set to a list of subjects for dropdown
    subjects = [{'subject_name': subject} for subject in subjects_set]
    
    # Get selected date and subject from form (GET or POST)
    selected_date = request.form.get('selected_date') if request.method == 'POST' else None
    selected_subject_name = request.form.get('subject_code') if request.method == 'POST' else None
    
    # Initialize date_attendance to hold attendance data for the selected date and subject
    date_attendance = {student_data['name']: "Absent"
        for student_id, student_data in students.items()
        if selected_subject_name in student_data['subjects']}

    # Fetch attendance for the selected date and subject
    if selected_date and selected_subject_name:
        if selected_subject_name in attendance_log.get(selected_date, {}):
            # Update attendance with "Present" or any existing statuses from the log
            for student_name, status in attendance_log[selected_date][selected_subject_name].items():
                date_attendance[student_name] = status
    
    # Calculate total attendance for each subject
    total_subject_attendance = {}
    for record in attendance_records:
        student_name = record.get('student_name')
        subject_name = record.get('subject_name')
        status = record.get('status', "Absent")
        
        if not student_name or not subject_name:
            continue
        
        # Initialize the subject in the dictionary if not already present
        if subject_name not in total_subject_attendance:
            total_subject_attendance[subject_name] = {}

        # Initialize the student in the subject-specific dictionary if not already present
        if student_name not in total_subject_attendance[subject_name]:
            total_subject_attendance[subject_name][student_name] = 0
        
        # Increment count if status is "Present"
        if status == "Present":
            total_subject_attendance[subject_name][student_name] += 1

    # Return the correct context for the template
    return render_template(
        'view_attendance.html', 
        students=students, 
        selected_date=selected_date,
        selected_subject_name=selected_subject_name,
        total_subject_attendance=total_subject_attendance,  # Total attendance by subject
        subjects=subjects,  # List of subjects for dropdown
        date_attendance=date_attendance  # Attendance data for the selected subject and date
    )



def clean_attendance_log():
    # Fetch all attendance records from MongoDB
    for record in attendance_log_collection.find():  # Correct MongoDB collection query
        date = record.get('date')
        daily_log = record.get('daily_log', None)

        if not daily_log:  # If daily_log is None or empty, skip the record
            print(f"Invalid or missing 'daily_log' for record with date {date}. Skipping...")
            continue
        
        # Loop through each student and mark attendance
        for student_name in daily_log:
            # Update attendance for each student, marking as Present
            attendance_log_collection.update_one(
                {"date": date},  # Filter by date
                {"$set": {f"attendance.{student_name}": "Present"}},  # Update attendance for the student
                upsert=True  # Ensure new documents are created if no record exists for the date
            )
    
    print("Attendance log cleaned successfully.")

from bson import ObjectId
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    try:
        if 'user' not in session:
            return redirect(url_for('login'))

        user_email = session['user']
        user = mongo.db.users.find_one({'email': user_email})
        if not user:
            flash('User not found. Please log in again.', 'error')
            return redirect(url_for('login'))

        admin_name = user.get('name', 'Admin')
        semesters = user.get('semesters', [])

        # Handle adding a subject
        if request.method == 'POST':
            subject_name = request.form.get('subject_name')
            subject_code = request.form.get('subject_code')
            semester_name = request.form.get('semester_name')

            if not subject_name or not subject_code or not semester_name:
                flash('Subject name, code, and semester are required.', 'error')
            else:
                subject_data = {
                    'subject_name': subject_name,
                    'subject_code': subject_code
                }

                # Find the correct semester and add the subject
                mongo.db.users.update_one(
                    {'email': user_email, 'semesters.name': semester_name},
                    {'$push': {'semesters.$.subjects': subject_data}}
                )

                flash(f'Subject "{subject_name}" added to {semester_name}!', 'success')

                # Re-fetch user data to update the semesters list
                user = mongo.db.users.find_one({'email': user_email})
                semesters = user.get('semesters', [])

        return render_template('dashboard.html', admin_name=admin_name, semesters=semesters)
    except Exception as e:
        print(f"Error in dashboard route: {e}")
        flash('An unexpected error occurred. Please try again later.', 'error')
        return redirect(url_for('login'))




if __name__ == "__main__":
    clean_attendance_log()
    socketio.run(app, debug=True)   