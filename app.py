from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import mysql.connector
from mysql.connector import pooling
import cv2
import numpy as np
import os
import uuid
from detect import LeukemiaDetector
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import re
import random
import logging
import traceback
import bleach

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: 'python-dotenv' not found. Using default values for environment variables.")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fixed_key_for_testing_123')
app.permanent_session_lifetime = timedelta(days=7)
app.config['SESSION_PERMANENT'] = True
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'Uploads')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Rate Limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Logging Setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
TEMP_IMAGE_DIR = "static/temp_images"
TEMP_REPORT_DIR = "static/temp_reports"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_REPORT_DIR, exist_ok=True)

# Model Initialization
MODEL_PATH = "model/s1vgg16Unet_blood_cancer_multitask_final.h5"
detector = None
try:
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}. Please ensure the model file exists.")
    else:
        logger.debug(f"Loading model from: {MODEL_PATH}")
        detector = LeukemiaDetector(MODEL_PATH)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    detector = None

# Database Connection Pool
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "2003",
    "database": "blood_cancer",
    "pool_name": "mypool",
    "pool_size": 5,
    "pool_reset_session": True
}

try:
    connection_pool = pooling.MySQLConnectionPool(**db_config)
    logger.info("Database connection pool initialized successfully")
except mysql.connector.Error as e:
    logger.error(f"Failed to initialize database connection pool: {e}")
    connection_pool = None

def get_db_connection():
    try:
        conn = connection_pool.get_connection()
        if conn.is_connected():
            return conn
        else:
            logger.error("Database connection is not established")
            return None
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def initialize_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SHOW COLUMNS FROM doctors LIKE 'diagnosis_count'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE doctors ADD COLUMN diagnosis_count INT DEFAULT 0")
                conn.commit()
                logger.info("Added diagnosis_count column to doctors table")
        except mysql.connector.Error as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

initialize_database()

# Email Configuration
EMAIL_USER = os.getenv('EMAIL_USER', 'bloodcancerdiagnosis@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'hbnvrhgszxrpoktm')

def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Email authentication failed: {e}. Check EMAIL_PASSWORD (use App Password for Gmail 2FA).")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

# Validation Functions
def validate_aadhaar(aadhaar):
    return bool(re.match(r'^\d{12}$', aadhaar))

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def validate_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))

def validate_phone(phone):
    return bool(re.match(r'^\d{10,15}$', phone))

def generate_otp():
    return str(random.randint(100000, 999999))

# Sanitize Input
def sanitize_input(value):
    if isinstance(value, str):
        return bleach.clean(value, strip=True)
    return value

def convert_numpy_types(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Routes
@app.route('/static/uploads/<path:filename>')
def serve_uploaded_file(filename):
    filename = sanitize_input(filename)
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        logger.error(f"Failed to serve file {filename}: {e}")
        return "File not found", 404

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    email = sanitize_input(request.form.get('email', '').strip())
    message = sanitize_input(request.form.get('message', '').strip())
    timestamp = datetime.now()
    if not validate_email(email):
        flash("Invalid email format.", "error")
        return jsonify({'success': False, 'message': 'Invalid email format'})
    if not message:
        flash("Message cannot be empty.", "error")
        return jsonify({'success': False, 'message': 'Message cannot be empty'})
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO messages (email, message, timestamp) VALUES (%s, %s, %s)", (email, message, timestamp))
            conn.commit()
            logger.info(f"Message saved from {email}")
            return jsonify({'success': True, 'message': 'Message saved successfully'})
        except mysql.connector.Error as e:
            logger.error(f"Error saving message: {e}")
            flash(f"Error saving message: {str(e)}", "error")
            conn.rollback()
            return jsonify({'success': False, 'message': f'Error saving message: {str(e)}'})
        finally:
            cursor.close()
            conn.close()
    else:
        flash("Database connection failed", "error")
        return jsonify({'success': False, 'message': 'Database connection failed'})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/doctor/about')
def doctor_about():
    if 'doctor_id' not in session:
        flash("Please log in to access the doctor about page", "info")
        return redirect(url_for('doctor_login'))
    return render_template('doctor_about.html', doctor_username=session['doctor_username'])

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/diagnosis')
def diagnosis():
    if 'doctor_id' not in session:
        flash("Please log in to access the diagnosis page.", "error")
        return redirect(url_for('doctor_login'))
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'doctor_id' not in session:
        flash("Please log in to access the diagnosis page.", "error")
        return redirect(url_for('doctor_login'))
    if 'image' not in request.files:
        logger.error("No image uploaded in /analyze")
        flash('No image uploaded')
        return redirect(url_for('diagnosis'))
    if detector is None:
        logger.error("Model not initialized in /analyze")
        flash(f'Model initialization failed. Please ensure the model file exists at {MODEL_PATH} or contact the administrator.')
        return redirect(url_for('diagnosis'))
    image_file = request.files['image']
    if image_file.filename == '':
        logger.error("No image selected in /analyze")
        flash('No image selected')
        return redirect(url_for('diagnosis'))
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not '.' in image_file.filename or image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.')
        return redirect(url_for('diagnosis'))
    try:
        logger.info("Reading uploaded image")
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode uploaded image")
            flash('Invalid image format. Please upload a valid PNG or JPEG image.')
            return redirect(url_for('diagnosis'))
        logger.info(f"Performing classification on image of shape: {image.shape}")
        original_image, cancer_marking, prediction, confidence, cancer_count = detector.detect_cancer(image, perform_segmentation=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = str(uuid.uuid4())
        image_filename = f"image_{timestamp}_{result_id}.png"
        original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        if not cv2.imwrite(original_img_path, original_image):
            logger.error("Failed to save original image")
            flash('Failed to save image. Please try again.')
            return redirect(url_for('diagnosis'))
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("UPDATE doctors SET diagnosis_count = diagnosis_count + 1 WHERE id = %s", (session['doctor_id'],))
                conn.commit()
                logger.info(f"Updated diagnosis count for doctor ID: {session['doctor_id']}")
            except mysql.connector.Error as e:
                logger.error(f"Error updating diagnosis count: {e}")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()
        else:
            flash("Database connection failed", "error")
        result_data = {
            'result_id': result_id,
            'original_img_path': f"/static/Uploads/{image_filename}",
            'prediction': prediction,
            'confidence': confidence,
            'cancer_count': cancer_count,
            'cancer_marking_path': None,
            'advanced_analysis': False,
            'image_filename': image_filename
        }
        result_data = convert_numpy_types(result_data)
        logger.debug(f"Converted NumPy types in result_data: {result_data}")
        session['result_data'] = result_data
        session['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Rendering result.html with data: {result_data}")
        return render_template('result.html', result_data=result_data)
    except Exception as e:
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        flash(f'Analysis failed: {str(e)}. Please try again or contact support.')
        return redirect(url_for('diagnosis'))

@app.route('/advanced_analyze', methods=['POST'])
def advanced_analyze():
    if 'doctor_id' not in session:
        flash("Please log in to access the diagnosis page.", "error")
        return redirect(url_for('doctor_login'))
    if detector is None:
        logger.error("Model not initialized in /advanced_analyze")
        flash(f'Model initialization failed. Please ensure the model file exists at {MODEL_PATH} or contact the administrator.')
        return redirect(url_for('diagnosis'))
    try:
        result_data = session.get('result_data')
        if not result_data or 'image_filename' not in result_data:
            logger.error("No previous analysis found in /advanced_analyze")
            flash('No previous analysis found.')
            return redirect(url_for('diagnosis'))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_data['image_filename'])
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image for advanced analysis")
            flash('Failed to load previous image.')
            return redirect(url_for('diagnosis'))
        logger.info(f"Performing advanced detection on image of shape: {image.shape}")
        original_image, cancer_marking, prediction, confidence, cancer_count = detector.detect_cancer(image, perform_segmentation=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = result_data['result_id']
        cancer_marking_filename = None
        original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_data['image_filename'])
        if not cv2.imwrite(original_img_path, original_image):
            logger.error("Failed to save original image")
            flash('Failed to save image. Please try again.')
            return redirect(url_for('diagnosis'))
        if prediction not in ["Benign", "Invalid image", "Error", "Uncertain"] and cancer_marking is not None:
            cancer_marking_filename = f"marked_{timestamp}_{result_id}.png"
            cancer_marking_path = os.path.join(app.config['UPLOAD_FOLDER'], cancer_marking_filename)
            if not cv2.imwrite(cancer_marking_path, cancer_marking):
                logger.error("Failed to save cancer marking image")
                flash('Failed to save analysis image. Please try again.')
                return redirect(url_for('diagnosis'))
        updated_result_data = {
            'result_id': result_id,
            'original_img_path': f"/static/Uploads/{result_data['image_filename']}",
            'prediction': prediction,
            'confidence': confidence,
            'cancer_count': cancer_count,
            'cancer_marking_path': f"/static/Uploads/{cancer_marking_filename}" if cancer_marking_filename else None,
            'advanced_analysis': True,
            'image_filename': result_data['image_filename']
        }
        updated_result_data = convert_numpy_types(updated_result_data)
        logger.debug(f"Converted NumPy types in updated_result_data: {updated_result_data}")
        session['result_data'] = updated_result_data
        session['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Rendering result.html with advanced data: {updated_result_data}")
        return render_template('result.html', result_data=updated_result_data)
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}\n{traceback.format_exc()}")
        flash(f'Advanced analysis failed: {str(e)}. Please try again or contact support.')
        return redirect(url_for('diagnosis'))

@app.route('/result')
def result():
    if 'doctor_id' not in session:
        logger.error("Redirect to doctor_login: doctor_id not in session")
        flash("Please log in to access the result page.", "error")
        return redirect(url_for('doctor_login'))
    if 'result_data' not in session:
        logger.error("Redirect to diagnosis: result_data not in session")
        flash("No result data found. Please upload an image for diagnosis.", "error")
        return redirect(url_for('diagnosis'))
    data = session['result_data']
    logger.info(f"Rendering result page with data: {data}")
    return render_template('result.html', result_data=data)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'doctor_id' not in session:
        logger.error("Redirect to doctor_login: doctor_id not in session")
        flash("Please log in to access the report generation.", "error")
        return redirect(url_for('doctor_login'))

    name = sanitize_input(request.form.get('name', '').strip())
    age = sanitize_input(request.form.get('age', '').strip())
    gender = sanitize_input(request.form.get('gender', '').strip())
    email = sanitize_input(request.form.get('email', '').strip())
    phone = sanitize_input(request.form.get('phone', '').strip())
    address = sanitize_input(request.form.get('address', '').strip())
    diagnosis = sanitize_input(request.form.get('diagnosis', '').strip())
    confidence = sanitize_input(request.form.get('confidence', '').strip())

    if not all([name, age, gender, email, phone, address, diagnosis, confidence]):
        logger.error("Missing fields in generate_report")
        flash("All fields are required to generate a report.", "error")
        return redirect(url_for('patients'))

    if diagnosis == "Uncertain":
        logger.error("Attempted to generate report for uncertain prediction")
        flash("Cannot generate report for uncertain predictions.", "error")
        return redirect(url_for('patients'))

    try:
        age = int(age)
        if age < 0 or age > 150:
            flash("Invalid age. Age must be between 0 and 150.", "error")
            return redirect(url_for('patients'))
    except ValueError:
        flash("Invalid age. Age must be a number.", "error")
        return redirect(url_for('patients'))

    if not validate_email(email):
        flash("Invalid email format.", "error")
        return redirect(url_for('patients'))

    if not validate_phone(phone):
        flash("Invalid phone number. Must be 10-15 digits.", "error")
        return redirect(url_for('patients'))

    if gender not in ['Male', 'Female', 'Other']:
        flash("Invalid gender selection.", "error")
        return redirect(url_for('patients'))

    doctor_id = session['doctor_id']
    doctor_name = session.get('doctor_username', 'Unknown Doctor')
    timestamp = session.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_id = session.get('result_data', {}).get('result_id', str(uuid.uuid4()))

    try:
        os.makedirs(TEMP_REPORT_DIR, exist_ok=True)
        if not os.access(TEMP_REPORT_DIR, os.W_OK):
            logger.error(f"No write permission for directory: {TEMP_REPORT_DIR}")
            flash("Server error: No write permission for report directory.", "error")
            return redirect(url_for('patients'))

        report_path = os.path.normpath(os.path.join(TEMP_REPORT_DIR, f"report_{result_id}.pdf"))
        logger.info(f"Generating report at: {report_path}")

        pdf = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Blood Cancer Detection Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Doctor Details", styles['Heading2']))
        story.append(Paragraph(f"Doctor ID: {doctor_id}", styles['Normal']))
        story.append(Paragraph(f"Doctor Name: {doctor_name}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Patient Details", styles['Heading2']))
        story.append(Paragraph(f"Name: {name}", styles['Normal']))
        story.append(Paragraph(f"Age: {age}", styles['Normal']))
        story.append(Paragraph(f"Gender: {gender}", styles['Normal']))
        story.append(Paragraph(f"Email: {email}", styles['Normal']))
        story.append(Paragraph(f"Phone: {phone}", styles['Normal']))
        story.append(Paragraph(f"Address: {address}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Detection Results", styles['Heading2']))
        story.append(Paragraph(f"Cancer Detected: {diagnosis}", styles['Normal']))
        story.append(Paragraph(f"Confidence: {confidence}", styles['Normal']))
        if diagnosis == "Benign":
            story.append(Paragraph("Note: No cancer detected in this sample.", styles['Normal']))
        story.append(Paragraph(f"Date & Time: {timestamp}", styles['Normal']))

        logger.debug("Building PDF document")
        pdf.build(story)
        logger.debug(f"PDF generation completed for: {report_path}")

        if not os.path.exists(report_path):
            logger.error(f"Report file was not created at: {report_path}")
            flash("Failed to create report file. Please try again.", "error")
            return redirect(url_for('patients'))

        logger.info(f"Report generated and verified: {report_path}")
        session.pop('result_data', None)
        session.pop('timestamp', None)

        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"report_{name}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}\n{traceback.format_exc()}")
        flash(f"Error generating report: {str(e)}", "error")
        return redirect(url_for('patients'))

@app.route('/patients', methods=['GET', 'POST'])
def patients():
    if 'doctor_id' not in session:
        logger.error("Redirect to doctor_login: doctor_id not in session")
        flash("Please log in to access the patients page.", "error")
        return redirect(url_for('doctor_login'))

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patients WHERE doctor_id = %s ORDER BY timestamp DESC", (session['doctor_id'],))
        all_patients = cursor.fetchall()

        if request.method == 'POST':
            action = request.form.get('action')
            name = sanitize_input(request.form.get('name', '').strip())
            age = sanitize_input(request.form.get('age', '').strip())
            gender = sanitize_input(request.form.get('gender', '').strip())
            email = sanitize_input(request.form.get('email', '').strip())
            phone = sanitize_input(request.form.get('phone', '').strip())
            address = sanitize_input(request.form.get('address', '').strip())

            if not all([name, age, gender, email, phone, address]):
                logger.error("Missing fields in patients form")
                flash("All fields are required.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('patients'))

            try:
                age = int(age)
                if age < 0 or age > 150:
                    flash("Invalid age. Age must be between 0 and 150.", "error")
                    cursor.close()
                    conn.close()
                    return redirect(url_for('patients'))
            except ValueError:
                flash("Invalid age. Age must be a number.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('patients'))

            if not validate_email(email):
                flash("Invalid email format.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('patients'))

            if not validate_phone(phone):
                flash("Invalid phone number. Must be 10-15 digits.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('patients'))

            if gender not in ['Male', 'Female', 'Other']:
                flash("Invalid gender selection.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('patients'))

            result_data = session.get('result_data', {})
            prediction = result_data.get('prediction', 'N/A')
            confidence = result_data.get('confidence', 0.0)
            result_id = result_data.get('result_id', str(uuid.uuid4()))
            timestamp = session.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            doctor_id = session['doctor_id']
            doctor_name = session.get('doctor_username', 'Unknown Doctor')

            if action == 'save':
                try:
                    cursor.execute("""
                        INSERT INTO patients (id, timestamp, name, age, gender, email, phone, address, diagnosis, confidence, doctor_id, doctor_name)
                        VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (timestamp, name, age, gender, email, phone, address, prediction, confidence, doctor_id, doctor_name))
                    conn.commit()
                    flash("Patient details saved successfully!", "success")
                    logger.info(f"Saved patient data for {name}")
                except Exception as e:
                    logger.error(f"Error saving patient data: {e}\n{traceback.format_exc()}")
                    flash(f"Error saving patient data: {str(e)}", "error")
                    conn.rollback()
                finally:
                    cursor.close()
                    conn.close()
                return redirect(url_for('patients'))

            elif action == 'generate':
                logger.info("Redirecting to generate_report from patients")
                return redirect(url_for('generate_report'))

        cursor.close()
        conn.close()
    else:
        flash("Database connection failed", "error")
        all_patients = []

    return render_template('patients.html', patients=all_patients, result_data=session.get('result_data', {}), timestamp=session.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

@app.route('/download_report/<int:patient_id>')
def download_report(patient_id):
    if 'doctor_id' not in session:
        logger.error("Redirect to doctor_login: doctor_id not in session")
        flash("Please log in to access the report download.", "error")
        return redirect(url_for('doctor_login'))

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM doctors WHERE id = %s", (session['doctor_id'],))
        doctor = cursor.fetchone()
        if doctor and doctor['first_login']:
            flash("Please change your password before accessing this page.", "info")
            cursor.close()
            conn.close()
            return redirect(url_for('doctor_dashboard'))

        cursor.execute("SELECT * FROM patients WHERE id = %s AND doctor_id = %s", (patient_id, session['doctor_id']))
        patient = cursor.fetchone()
        cursor.close()
        conn.close()
    else:
        flash("Database connection failed", "error")
        return redirect(url_for('patients'))

    if not patient:
        flash("Patient not found or not authorized.", "error")
        return redirect(url_for('patients'))

    report_path = os.path.normpath(os.path.join(TEMP_REPORT_DIR, f"report_{patient['id']}.pdf"))
    if not os.path.exists(report_path):
        logger.info(f"Generating report for patient ID {patient['id']}: {report_path}")
        try:
            os.makedirs(TEMP_REPORT_DIR, exist_ok=True)
            if not os.access(TEMP_REPORT_DIR, os.W_OK):
                logger.error(f"No write permission for directory: {TEMP_REPORT_DIR}")
                flash("Server error: No write permission for report directory.", "error")
                return redirect(url_for('patients'))

            pdf = SimpleDocTemplate(report_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("Blood Cancer Detection Report", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Doctor Details", styles['Heading2']))
            story.append(Paragraph(f"Doctor ID: {patient['doctor_id']}", styles['Normal']))
            story.append(Paragraph(f"Doctor Name: {patient['doctor_name']}", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Patient Details", styles['Heading2']))
            story.append(Paragraph(f"Name: {patient['name']}", styles['Normal']))
            story.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
            story.append(Paragraph(f"Gender: {patient['gender']}", styles['Normal']))
            story.append(Paragraph(f"Email: {patient['email']}", styles['Normal']))
            story.append(Paragraph(f"Phone: {patient['phone']}", styles['Normal']))
            story.append(Paragraph(f"Address: {patient['address']}", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Detection Results", styles['Heading2']))
            story.append(Paragraph(f"Cancer Detected: {patient['diagnosis']}", styles['Normal']))
            story.append(Paragraph(f"Confidence: {patient['confidence']:.2%}", styles['Normal']))
            if patient['diagnosis'] == "Benign":
                story.append(Paragraph("Note: No cancer detected in this sample.", styles['Normal']))
            story.append(Paragraph(f"Date & Time: {patient['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

            pdf.build(story)
            logger.info(f"Report generated: {report_path}")

            if not os.path.exists(report_path):
                logger.error(f"Report file was not created at: {report_path}")
                flash("Failed to create report file. Please try again.", "error")
                return redirect(url_for('patients'))
        except Exception as e:
            logger.error(f"Error generating report: {e}\n{traceback.format_exc()}")
            flash(f"Error generating report: {str(e)}", "error")
            return redirect(url_for('patients'))

    try:
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"report_{patient['name']}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error sending report file: {e}\n{traceback.format_exc()}")
        flash("Error downloading report. Please try again.", "error")
        return redirect(url_for('patients'))

@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def admin_login():
    if request.method == 'POST':
        username = sanitize_input(request.form.get('username', 'admin').strip())
        logger.info(f"Admin login attempt - Username: {username}")
        session.clear()
        session['admin'] = username
        session.permanent = True
        flash("Login successful!", "success")
        return redirect(url_for('admin_dashboard'))
    else:
        if 'admin' not in session:
            flash("Please log in to access the admin dashboard", "info")
    return render_template('admin_login.html')

@app.route('/doctor/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def doctor_login():
    if request.method == 'POST':
        username = sanitize_input(request.form.get('username', '').strip())
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template('doctor_login.html')
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute("SELECT * FROM doctors WHERE doctor_id = %s OR gmail = %s OR mobile_number = %s", (username, username, username))
                doctor = cursor.fetchone()
                if doctor and not doctor['is_blocked'] and check_password_hash(doctor['password'], password):
                    session.clear()
                    session['doctor_id'] = doctor['id']
                    session['doctor_username'] = doctor['doctor_name']
                    session.permanent = True
                    session.modified = True
                    flash("Login successful!", "success")
                    logger.info(f"Doctor login successful: {doctor['doctor_name']}")
                    return redirect(url_for('doctor_dashboard'))
                elif doctor and doctor['is_blocked']:
                    flash("Your account has been blocked by the admin. Contact support for assistance.", "error")
                    logger.warning(f"Blocked doctor login attempt: {username}")
                else:
                    flash("Invalid doctor credentials", "error")
                    logger.warning(f"Failed doctor login attempt: {username}")
            except mysql.connector.Error as e:
                logger.error(f"Database query error during doctor login: {e}")
                flash("Database error occurred", "error")
            finally:
                cursor.close()
                conn.close()
        else:
            flash("Database connection failed", "error")
    return render_template('doctor_login.html')

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'doctor_id' not in session:
        flash("Please log in to access the dashboard.", "error")
        return redirect(url_for('doctor_login'))
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT doctor_name, first_login FROM doctors WHERE id = %s", (session['doctor_id'],))
            doctor = cursor.fetchone()
            if doctor:
                doctor_username = doctor['doctor_name']
                first_login = doctor['first_login']
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return render_template('doctor_dashboard.html', doctor_username=doctor_username, first_login=first_login, timestamp=timestamp)
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            flash("An error occurred while fetching dashboard data.", "error")
        finally:
            cursor.close()
            conn.close()
    flash("Database connection failed.", "error")
    return redirect(url_for('doctor_login'))

@app.route('/doctor/profile')
def doctor_profile():
    if 'doctor_id' not in session:
        flash("Please log in to access your profile", "info")
        return redirect(url_for('doctor_login'))
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM doctors WHERE id = %s", (session['doctor_id'],))
            doctor = cursor.fetchone()
            if not doctor:
                flash("Doctor not found", "error")
                return redirect(url_for('doctor_login'))
            if doctor['first_login']:
                flash("Please change your password before accessing this page.", "info")
                return redirect(url_for('doctor_dashboard'))
            diagnosis_count = doctor['diagnosis_count'] if doctor['diagnosis_count'] is not None else 0
            profile_data = {
                'doctor_id': doctor['doctor_id'],
                'doctor_name': doctor['doctor_name'],
                'age': doctor['age'],
                'dob': doctor['dob'].strftime('%Y-%m-%d') if doctor['dob'] else 'N/A',
                'mobile_number': doctor['mobile_number'],
                'gmail': doctor['gmail'],
                'aadhaar_number': doctor['aadhaar_number'],
                'hospital_name': doctor['hospital_name'] or 'N/A',
                'hospital_address': doctor['hospital_address'] or 'N/A',
                'diagnosis_count': diagnosis_count,
                'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            flash("Database error occurred", "error")
            return redirect(url_for('doctor_login'))
        finally:
            cursor.close()
            conn.close()
    else:
        flash("Database connection failed", "error")
        return redirect(url_for('doctor_login'))
    return render_template('profile.html', **profile_data)

@app.route('/medication')
def medication():
    if 'doctor_id' not in session:
        return redirect(url_for('doctor_login'))
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT first_login FROM doctors WHERE id = %s", (session['doctor_id'],))
        doctor = cursor.fetchone()
        if doctor and doctor['first_login']:
            flash("Please change your password before accessing this page.", "info")
            return redirect(url_for('doctor_dashboard'))
        cursor.close()
        conn.close()
    else:
        flash("Database connection failed", "error")
        return redirect(url_for('doctor_login'))
    return render_template('medication.html', doctor_username=session['doctor_username'])

@app.route('/doctor/profile_data')
def doctor_profile_data():
    if 'doctor_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            doctor_id = session['doctor_id']
            cursor.execute("SELECT doctor_id, doctor_name, hospital_name, diagnosis_count FROM doctors WHERE id = %s", (doctor_id,))
            doctor = cursor.fetchone()
            if doctor:
                return jsonify({
                    'doctor_id': doctor['doctor_id'],
                    'doctor_name': doctor['doctor_name'],
                    'hospital_name': doctor['hospital_name'] or 'N/A',
                    'diagnosis_count': doctor['diagnosis_count'] if doctor['diagnosis_count'] is not None else 0
                })
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'error': 'Database connection failed'}), 500

@app.route('/doctor/change_password', methods=['POST'])
def change_password():
    if 'doctor_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'Invalid request data'}), 400
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')
    if not current_password or not new_password or not confirm_password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    if new_password != confirm_password:
        return jsonify({'success': False, 'message': 'New password and confirmation do not match'}), 400
    is_valid, message = validate_password(new_password)
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            doctor_id = session['doctor_id']
            cursor.execute("SELECT password, doctor_name, gmail FROM doctors WHERE id = %s", (doctor_id,))
            doctor = cursor.fetchone()
            if doctor and check_password_hash(doctor['password'], current_password):
                hashed_password = generate_password_hash(new_password)
                cursor.execute("UPDATE doctors SET password = %s, first_login = %s WHERE id = %s", (hashed_password, False, doctor_id))
                conn.commit()
                logger.info(f"Password changed successfully for doctor: {doctor['doctor_name']}")
                subject = "Password Changed Successfully"
                body = f"Dear {doctor['doctor_name']},\n\nYour password has been changed successfully.\n\nIf you did not initiate this change, please contact support immediately.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                send_email(doctor['gmail'], subject, body)
                return jsonify({'success': True, 'message': 'Password changed successfully'})
            else:
                return jsonify({'success': False, 'message': 'Incorrect current password'}), 400
        except mysql.connector.Error as e:
            logger.error(f"Database error during password change: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'success': False, 'message': 'Database connection failed'}), 500

@app.route('/doctor/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        data = request.get_json()
        email = sanitize_input(data.get('email', '').strip())
        if not email:
            return jsonify({'success': False, 'message': 'Email is required'}), 400
        if not validate_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400
        return jsonify({'success': True, 'message': 'Proceed to send OTP'})
    return render_template('forgot_password.html')

@app.route('/doctor/send_otp', methods=['POST'])
@limiter.limit("5 per minute")
def send_otp():
    data = request.get_json()
    email = sanitize_input(data.get('email', '').strip())
    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400
    if not validate_email(email):
        return jsonify({'success': False, 'message': 'Invalid email format'}), 400
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id, doctor_name FROM doctors WHERE gmail = %s", (email,))
            doctor = cursor.fetchone()
            if doctor:
                otp = generate_otp()
                session['forgot_otp'] = otp
                session['forgot_email'] = email
                session['forgot_otp_timestamp'] = datetime.now().timestamp()
                session['forgot_doctor_id'] = doctor['id']
                subject = "Your OTP for Password Reset"
                body = f"Dear {doctor['doctor_name']},\n\nYour OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                if send_email(email, subject, body):
                    logger.info(f"OTP sent to {email} for password reset")
                    return jsonify({'success': True, 'message': 'OTP sent successfully'})
                else:
                    logger.error(f"Failed to send OTP email to {email}")
                    return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
            else:
                return jsonify({'success': False, 'message': 'Email not found'}), 400
        except mysql.connector.Error as e:
            logger.error(f"Database error during OTP send: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'success': False, 'message': 'Database connection failed'}), 500

@app.route('/doctor/verify_otp', methods=['POST'])
@limiter.limit("5 per minute")
def verify_otp():
    data = request.get_json()
    otp = sanitize_input(data.get('otp', '').strip())
    if not otp:
        return jsonify({'success': False, 'message': 'OTP is required'}), 400
    if 'forgot_otp' not in session or 'forgot_otp_timestamp' not in session:
        return jsonify({'success': False, 'message': 'No OTP session found'}), 400
    current_time = datetime.now().timestamp()
    otp_time = session['forgot_otp_timestamp']
    if (current_time - otp_time) > 600:
        session.pop('forgot_otp', None)
        session.pop('forgot_email', None)
        session.pop('forgot_otp_timestamp', None)
        session.pop('forgot_doctor_id', None)
        return jsonify({'success': False, 'message': 'OTP has expired'}), 400
    if session['forgot_otp'] == otp:
        session['forgot_otp_verified'] = True
        logger.info(f"OTP verified for email: {session['forgot_email']}")
        return jsonify({'success': True, 'message': 'OTP verified successfully'})
    else:
        logger.warning(f"Invalid OTP attempt for email: {session['forgot_email']}")
        return jsonify({'success': False, 'message': 'Invalid OTP'}), 400

@app.route('/doctor/resend_otp', methods=['POST'])
@limiter.limit("3 per minute")
def resend_otp():
    data = request.get_json()
    email = sanitize_input(data.get('email', '').strip())
    if not email or 'forgot_email' not in session or email != session['forgot_email']:
        return jsonify({'success': False, 'message': 'Invalid email or session'}), 400
    if not validate_email(email):
        return jsonify({'success': False, 'message': 'Invalid email format'}), 400
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT doctor_name FROM doctors WHERE gmail = %s", (email,))
            doctor = cursor.fetchone()
            if doctor:
                otp = generate_otp()
                session['forgot_otp'] = otp
                session['forgot_otp_timestamp'] = datetime.now().timestamp()
                subject = "Your New OTP for Password Reset"
                body = f"Dear {doctor['doctor_name']},\n\nYour new OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                if send_email(email, subject, body):
                    logger.info(f"OTP resent to {email}")
                    return jsonify({'success': True, 'message': 'OTP resent successfully'})
                else:
                    logger.error(f"Failed to resend OTP email to {email}")
                    return jsonify({'success': False, 'message': 'Failed to resend OTP email'}), 500
            else:
                return jsonify({'success': False, 'message': 'Email not found'}), 400
        except mysql.connector.Error as e:
            logger.error(f"Database error during OTP resend: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'success': False, 'message': 'Database connection failed'}), 500

@app.route('/doctor/update_password', methods=['POST'])
def update_password():
    if 'forgot_otp_verified' not in session or not session['forgot_otp_verified']:
        return jsonify({'success': False, 'message': 'OTP verification required'}), 401
    data = request.get_json()
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')
    if not new_password or not confirm_password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    if new_password != confirm_password:
        return jsonify({'success': False, 'message': 'New password and confirmation do not match'}), 400
    is_valid, message = validate_password(new_password)
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            doctor_id = session['forgot_doctor_id']
            hashed_password = generate_password_hash(new_password)
            cursor.execute("UPDATE doctors SET password = %s, first_login = %s WHERE id = %s", (hashed_password, False, doctor_id))
            conn.commit()
            logger.info(f"Password updated via forgot password for doctor ID: {doctor_id}")
            email = session['forgot_email']
            cursor.execute("SELECT doctor_name FROM doctors WHERE id = %s", (doctor_id,))
            doctor = cursor.fetchone()
            if doctor:
                subject = "Password Reset Successful"
                body = f"Dear {doctor['doctor_name']},\n\nYour password has been reset successfully.\n\nIf you did not initiate this change, please contact support immediately.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                send_email(email, subject, body)
            session.pop('forgot_otp', None)
            session.pop('forgot_email', None)
            session.pop('forgot_otp_timestamp', None)
            session.pop('forgot_otp_verified', None)
            session.pop('forgot_doctor_id', None)
            return jsonify({'success': True, 'message': 'Password updated successfully'})
        except mysql.connector.Error as e:
            logger.error(f"Database error during password update: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'success': False, 'message': 'Database connection failed'}), 500

@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin' not in session:
        logger.warning("No admin session detected, redirecting to login")
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    doctors = []
    messages_data = []
    performance_data = []
    message_count = 0
    current_date = datetime.now()
    search_query = sanitize_input(request.args.get('search', '').strip()) if request.method == 'GET' else ''
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            if request.method == 'POST':
                if 'doctor_id' in request.form:
                    doctor_id = sanitize_input(request.form.get('doctor_id', '').strip())
                    doctor_name = sanitize_input(request.form.get('doctor_name', '').strip())
                    age = sanitize_input(request.form.get('age', '').strip())
                    dob = sanitize_input(request.form.get('dob', '').strip())
                    mobile_number = sanitize_input(request.form.get('mobile_number', '').strip())
                    gmail = sanitize_input(request.form.get('gmail', '').strip())
                    aadhaar_number = sanitize_input(request.form.get('aadhaar_number', '').strip())
                    hospital_name = sanitize_input(request.form.get('hospital_name', '').strip())
                    hospital_address = sanitize_input(request.form.get('hospital_address', '').strip())
                    password = request.form.get('password', '').strip()
                    if not all([doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address, password]):
                        flash("All fields are required.", "error")
                        return redirect(url_for('admin_dashboard'))
                    if not validate_aadhaar(aadhaar_number):
                        flash("Aadhaar number must be 12 digits.", "error")
                        return redirect(url_for('admin_dashboard'))
                    try:
                        age = int(age)
                        if age < 18 or age > 100:
                            flash("Invalid age. Age must be between 18 and 100.", "error")
                            return redirect(url_for('admin_dashboard'))
                    except ValueError:
                        flash("Invalid age. Age must be a number.", "error")
                        return redirect(url_for('admin_dashboard'))
                    if not validate_email(gmail):
                        flash("Invalid email format.", "error")
                        return redirect(url_for('admin_dashboard'))
                    if not validate_phone(mobile_number):
                        flash("Invalid phone number. Must be 10-15 digits.", "error")
                        return redirect(url_for('admin_dashboard'))
                    is_valid, message = validate_password(password)
                    if not is_valid:
                        flash(message, "error")
                        return redirect(url_for('admin_dashboard'))
                    cursor.execute("SELECT id FROM doctors WHERE doctor_id = %s", (doctor_id,))
                    if cursor.fetchone():
                        flash("Doctor ID already exists.", "error")
                        return redirect(url_for('admin_dashboard'))
                    cursor.execute("SELECT id FROM doctors WHERE aadhaar_number = %s", (aadhaar_number,))
                    if cursor.fetchone():
                        flash("Aadhaar number already exists.", "error")
                        return redirect(url_for('admin_dashboard'))
                    hashed_pw = generate_password_hash(password)
                    cursor.execute("""
                        INSERT INTO doctors (doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, 
                                          hospital_name, hospital_address, password, is_blocked, created_at, first_login)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, 
                          hospital_name, hospital_address, hashed_pw, 0, datetime.now(), True))
                    conn.commit()
                    flash(f"Doctor {doctor_name} created successfully!", "success")
                    subject = "Welcome to Blood Cancer Diagnosis - Account Created"
                    body = f"Dear {doctor_name},\n\nYour account has been created!\nUsername: {doctor_id}\nPassword: {password}\nLogin at http://127.0.0.1:5000/doctor/login.\n\nPlease change your password upon first login.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                    send_email(gmail, subject, body)
                    return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                elif 'toggle_id' in request.form:
                    doctor_id = request.form.get('toggle_id')
                    cursor.execute("SELECT is_blocked, doctor_name, gmail FROM doctors WHERE id = %s", (doctor_id,))
                    doctor = cursor.fetchone()
                    if doctor:
                        new_status = not doctor['is_blocked']
                        cursor.execute("UPDATE doctors SET is_blocked = %s WHERE id = %s", (int(new_status), doctor_id))
                        conn.commit()
                        flash(f"Doctor account {'blocked' if new_status else 'unblocked'} successfully!", "success")
                        subject = f"Account {'Blocked' if new_status else 'Unblocked'} Notification"
                        body = f"Dear {doctor['doctor_name']},\n\nYour account has been {'blocked' if new_status else 'unblocked'} by the admin.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                        send_email(doctor['gmail'], subject, body)
                    return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                elif 'edit_id' in request.form:
                    edit_id = request.form.get('edit_id')
                    doctor_id = sanitize_input(request.form.get('doctor_id', '').strip())
                    doctor_name = sanitize_input(request.form.get('doctor_name', '').strip())
                    age = sanitize_input(request.form.get('age', '').strip())
                    dob = sanitize_input(request.form.get('dob', '').strip())
                    mobile_number = sanitize_input(request.form.get('mobile_number', '').strip())
                    gmail = sanitize_input(request.form.get('gmail', '').strip())
                    aadhaar_number = sanitize_input(request.form.get('aadhaar_number', '').strip())
                    hospital_name = sanitize_input(request.form.get('hospital_name', '').strip())
                    hospital_address = sanitize_input(request.form.get('hospital_address', '').strip())
                    password = request.form.get('password', '').strip()
                    if not all([doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address]):
                        flash("All fields are required.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    if not validate_aadhaar(aadhaar_number):
                        flash("Aadhaar number must be 12 digits.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    try:
                        age = int(age)
                        if age < 18 or age > 100:
                            flash("Invalid age. Age must be between 18 and 100.", "error")
                            return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    except ValueError:
                        flash("Invalid age. Age must be a number.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    if not validate_email(gmail):
                        flash("Invalid email format.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    if not validate_phone(mobile_number):
                        flash("Invalid phone number. Must be 10-15 digits.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    hashed_pw = generate_password_hash(password) if password else None
                    update_query = """
                        UPDATE doctors 
                        SET doctor_id = %s, doctor_name = %s, age = %s, dob = %s, mobile_number = %s, gmail = %s, 
                            aadhaar_number = %s, hospital_name = %s, hospital_address = %s
                        WHERE id = %s
                    """
                    update_params = (doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address, edit_id)
                    if hashed_pw:
                        update_query += ", password = %s, first_login = %s"
                        update_params += (hashed_pw, True)
                    cursor.execute("SELECT id FROM doctors WHERE doctor_id = %s AND id != %s", (doctor_id, edit_id))
                    if cursor.fetchone():
                        flash("Doctor ID already exists.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    cursor.execute("SELECT id FROM doctors WHERE aadhaar_number = %s AND id != %s", (aadhaar_number, edit_id))
                    if cursor.fetchone():
                        flash("Aadhaar number already exists.", "error")
                        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
                    cursor.execute(update_query, update_params)
                    conn.commit()
                    flash("Doctor account updated successfully!", "success")
                    return redirect(url_for('admin_dashboard') + '#doctors-list-section')
            if search_query:
                query = """
                    SELECT * FROM doctors 
                    WHERE doctor_id LIKE %s 
                    OR doctor_name LIKE %s 
                    OR gmail LIKE %s 
                    OR mobile_number LIKE %s 
                    OR aadhaar_number LIKE %s
                """
                search_term = f"%{search_query}%"
                cursor.execute(query, (search_term, search_term, search_term, search_term, search_term))
                doctors = cursor.fetchall()
                if not doctors:
                    flash("Doctor is not valid.", "error")
            else:
                cursor.execute("SELECT * FROM doctors")
                doctors = cursor.fetchall()
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            message_count = cursor.fetchone()['count']
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC")
            messages_data = cursor.fetchall()
            cursor.execute("""
                SELECT d.doctor_id, d.doctor_name, d.diagnosis_count as patient_count
                FROM doctors d
                LEFT JOIN patients p ON d.doctor_id = p.doctor_id
                GROUP BY d.doctor_id, d.doctor_name, d.diagnosis_count
            """)
            performance_data = cursor.fetchall()
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            flash(f"Database error occurred: {str(e)}", "error")
        finally:
            cursor.close()
            conn.close()
    else:
        flash("Database connection failed", "error")
    return render_template('admin_dashboard.html', doctors=doctors, messages_data=messages_data, message_count=message_count, performance_data=performance_data, current_date=current_date, search_query=search_query)

@app.route('/admin/doctor/edit/<int:id>', methods=['GET', 'POST'])
def edit_doctor(id):
    if 'admin' not in session:
        flash("Please log in to access this page", "info")
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    if not conn:
        flash("Database connection failed", "error")
        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        doctor_id = sanitize_input(request.form.get('doctor_id', '').strip())
        doctor_name = sanitize_input(request.form.get('doctor_name', '').strip())
        age = sanitize_input(request.form.get('age', '').strip())
        dob = sanitize_input(request.form.get('dob', '').strip())
        mobile_number = sanitize_input(request.form.get('mobile_number', '').strip())
        gmail = sanitize_input(request.form.get('gmail', '').strip())
        aadhaar_number = sanitize_input(request.form.get('aadhaar_number', '').strip())
        hospital_name = sanitize_input(request.form.get('hospital_name', '').strip())
        hospital_address = sanitize_input(request.form.get('hospital_address', '').strip())
        password = request.form.get('password', '').strip()
        if not all([doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address]):
            flash("All fields are required.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        if not validate_aadhaar(aadhaar_number):
            flash("Aadhaar number must be 12 digits.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        try:
            age = int(age)
            if age < 18 or age > 100:
                flash("Invalid age. Age must be between 18 and 100.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for('edit_doctor', id=id))
        except ValueError:
            flash("Invalid age. Age must be a number.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        if not validate_email(gmail):
            flash("Invalid email format.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        if not validate_phone(mobile_number):
            flash("Invalid phone number. Must be 10-15 digits.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        if password:
            is_valid, message = validate_password(password)
            if not is_valid:
                flash(message, "error")
                cursor.close()
                conn.close()
                return redirect(url_for('edit_doctor', id=id))
        hashed_pw = generate_password_hash(password) if password else None
        update_query = """
            UPDATE doctors 
            SET doctor_id = %s, doctor_name = %s, age = %s, dob = %s, mobile_number = %s, gmail = %s, 
                aadhaar_number = %s, hospital_name = %s, hospital_address = %s
            WHERE id = %s
        """
        update_params = (doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address, id)
        if hashed_pw:
            update_query = update_query.rstrip() + ", password = %s, first_login = %s WHERE id = %s"
            update_params = (doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address, hashed_pw, True, id)
        cursor.execute("SELECT id FROM doctors WHERE doctor_id = %s AND id != %s", (doctor_id, id))
        if cursor.fetchone():
            flash("Doctor ID already exists.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        cursor.execute("SELECT id FROM doctors WHERE aadhaar_number = %s AND id != %s", (aadhaar_number, id))
        if cursor.fetchone():
            flash("Aadhaar number already exists.", "error")
            cursor.close()
            conn.close()
            return redirect(url_for('edit_doctor', id=id))
        try:
            cursor.execute(update_query, update_params)
            conn.commit()
            flash("Doctor account updated successfully!", "success")
        except mysql.connector.Error as e:
            logger.error(f"Error updating doctor: {str(e)}")
            flash(f"Error updating doctor: {str(e)}", "error")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
    cursor.execute("SELECT * FROM doctors WHERE id=%s", (id,))
    doctor = cursor.fetchone()
    cursor.close()
    conn.close()
    if not doctor:
        flash("Doctor not found.", "error")
        return redirect(url_for('admin_dashboard') + '#doctors-list-section')
    return render_template('edit_doctor.html', doctor=doctor)

@app.route('/admin/doctor/delete/<int:id>', methods=['POST'])
def delete_doctor(id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT doctor_name, gmail FROM doctors WHERE id=%s", (id,))
        doctor = cursor.fetchone()
        try:
            cursor.execute("DELETE FROM doctors WHERE id=%s", (id,))
            conn.commit()
            if doctor:
                flash(f"Doctor {doctor['doctor_name']} deleted successfully!", "success")
                subject = "Account Deletion Notification"
                body = f"Dear {doctor['doctor_name']},\n\nYour account has been deleted by the admin.\n\nDo not reply to this email.\nBest,\nBlood Cancer Diagnosis Team"
                send_email(doctor['gmail'], subject, body)
        except mysql.connector.Error as e:
            logger.error(f"Error deleting doctor: {str(e)}")
            flash(f"Error deleting doctor: {str(e)}", "error")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    else:
        flash("Database connection failed", "error")
    return redirect(url_for('admin_dashboard') + '#doctors-list-section')

@app.route('/admin/message/delete/<int:id>', methods=['POST'])
def delete_message(id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM messages WHERE id=%s", (id,))
            conn.commit()
            flash("Message deleted successfully!", "success")
        except mysql.connector.Error as e:
            logger.error(f"Error deleting message: {e}")
            flash(f"Error deleting message: {str(e)}", "error")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    else:
        flash("Database connection failed", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/check_duplicate')
def check_duplicate():
    if 'admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    field = request.args.get('field')
    value = sanitize_input(request.args.get('value', '').strip())
    if not field or not value or field not in ['doctor_id', 'aadhaar_number']:
        return jsonify({'error': 'Invalid field'}), 400
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            query = f"SELECT COUNT(*) FROM doctors WHERE {field} = %s"
            cursor.execute(query, (value,))
            count = cursor.fetchone()[0]
            return jsonify({'exists': count > 0})
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'error': 'Database connection failed'}), 500

@app.route('/admin/get_doctors')
def get_doctors():
    if 'admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id, doctor_id, doctor_name, age, dob, mobile_number, gmail, aadhaar_number, hospital_name, hospital_address, is_blocked FROM doctors")
            doctors = cursor.fetchall()
            return jsonify([
                {
                    'id': d['id'],
                    'doctor_id': d['doctor_id'] or 'N/A',
                    'doctor_name': d['doctor_name'] or 'N/A',
                    'age': d['age'] or 'N/A',
                    'dob': d['dob'].strftime('%Y-%m-%d') if d['dob'] else 'N/A',
                    'mobile_number': d['mobile_number'] or 'N/A',
                    'gmail': d['gmail'] or 'N/A',
                    'aadhaar_number': d['aadhaar_number'] or 'N/A',
                    'hospital_name': d['hospital_name'] or 'N/A',
                    'hospital_address': d['hospital_address'] or 'N/A',
                    'is_blocked': d['is_blocked']
                } for d in doctors
            ])
        except mysql.connector.Error as e:
            logger.error(f"Database query error: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            cursor.close()
            conn.close()
    return jsonify({'error': 'Database connection failed'}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)