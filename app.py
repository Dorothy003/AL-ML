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
        attendance_log.setdefault(date, {})[student_name] = 'Absent'
        save_attendance_log(attendance_log)
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
def index():
    if 'user' not in session:
        return redirect(url_for('login')) 
    return redirect(url_for('dashboard')) 
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

    return render_template('Addstudent.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists in the database
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            session['user'] = email
       
            return redirect(url_for('dashboard'))
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

        # Add 8 semesters to the new user
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


@app.route('/Attendancelog')
def Attendancelog():
    # Fetch students and attendance logs from MongoDB
    students_records = list(db.students.find({}))
    attendance_records = list(db.attendance.find({}))
    
    students = {}
    for record in students_records:
        student_id = str(record['_id'])  
        student_name = record.get('name', "Unknown")  # Default name if not found
        students[student_id] = {'name': student_name}
    
    # Initialize the attendance log
    attendance_log = {}
    for record in attendance_records:
        student_name = record.get('student_name')  
        date = record.get('date') 
        status = record.get('status', "Absent")  # Default to "Absent" if status is missing
        
        if not student_name or not date:
            continue  # Skip if student_name or date is missing
        
        if date not in attendance_log:
            attendance_log[date] = {}
        
        # Find the student_id by matching the name
        student_id = None
        for sid, student_data in students.items():
            if student_data['name'] == student_name:
                student_id = sid
                break
        
        if student_id:  # Ensure the student_id is found
            attendance_log[date][student_id] = status

    # Sort the dates
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
        subject = next((sub for sub in semester['subjects'] if sub['name'] == subject_name), None)
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
    students_records = list(db.students.find({}))
    attendance_records = list(db.attendance.find({}))
    
    # Create a dictionary to map student names and their IDs
    students = {}
    for record in students_records:
        student_id = str(record['_id'])
        student_name = record.get('name', "Unknown")
        students[student_id] = {'name': student_name}
    
    #  attendance_status dictionary
    attendance_log = {}
    for record in attendance_records:
        student_name = record.get('student_name')
        date = record.get('date')
        status = record.get('status', "Absent")
        
        if not student_name or not date:
            continue
        
        if date not in attendance_log:
            attendance_log[date] = {}
        
        attendance_log[date][student_name] = status

    # Get selected date from form
    selected_date = None
    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
    
    # Calculate total attendance count
    total_attendance_count = {}
    for date_data in attendance_log.values():
        for student_name, status in date_data.items():
            if status == "Present":
                total_attendance_count[student_name] = total_attendance_count.get(student_name, 0) + 1

    return render_template(
        'view_attendance.html', 
        students=students, 
        attendance_log=attendance_log, 
        dates=attendance_log.keys(), 
        selected_date=selected_date, 
        total_attendance_count=total_attendance_count
    )
def clean_attendance_log():
  
    for record in attendance_log_collection.find():  # Correct MongoDB collection query
        date = record.get('date')
        daily_log = record.get('daily_log', None)

        if not daily_log:  # If daily_log is None or empty, skip the record
            print(f"Invalid or missing 'log' for record with date {date}. Skipping...")
            continue
        
        # Loop through each student and mark attendance
        for student_name in daily_log:
            attendance_log_collection.update_one(
                {"date": date},  # Filter by date
                {"$set": {f"attendance.{student_name}": "Present"}},  # Update attendance for the student
                upsert=True 
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