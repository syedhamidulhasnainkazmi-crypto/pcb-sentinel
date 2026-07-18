"""
PCB Sentinel - Flask web app (No Database Version)
Works without Flask-SQLAlchemy - detection only, no history saving.
"""

import os
import json
import secrets
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, flash, send_from_directory, session
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import bcrypt

from detector import YOLOv8Detector, draw_detections

# ============ APP SETUP ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-secret-key-in-production')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ============ MODEL PATH ============
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

# ============ LOGIN MANAGER ============
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============ SIMPLE USER STORAGE (in-memory) ============
users = {}
user_scans = {}  # Track scans per user

FREE_TRIAL_SCANS = 5

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, company_name=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.company_name = company_name
        self.plan = 'trial'
        self.scans_used = 0
        self.is_verified = True  # Auto-verified for testing
    
    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def scans_remaining(self):
        return max(0, FREE_TRIAL_SCANS - self.scans_used)
    
    def can_scan(self):
        return self.scans_remaining() > 0

# ============ LOAD MODEL ============
detector = None
if os.path.exists(MODEL_PATH):
    try:
        detector = YOLOv8Detector(MODEL_PATH)
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
else:
    print(f"[WARN] Model file not found at {MODEL_PATH}")

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

# ============ PUBLIC ROUTES ============

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        company = request.form.get('company', '').strip()
        
        # Check if username exists
        for user in users.values():
            if user.username == username:
                flash('Username already taken.', 'error')
                return render_template('register.html')
            if user.email == email:
                flash('Email already registered.', 'error')
                return render_template('register.html')
        
        # Create user
        user_id = len(users) + 1
        new_user = User(user_id, username, email, '', company)
        new_user.set_password(password)
        users[user_id] = new_user
        user_scans[user_id] = 0
        
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        for user in users.values():
            if user.username == username and user.check_password(password):
                login_user(user)
                return redirect(url_for('dashboard'))
        
        flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/support')
def support():
    return render_template('support.html')

# ============ APP ROUTES ============

@app.route('/dashboard')
@login_required
def dashboard():
    scans_used = user_scans.get(current_user.id, 0)
    scans_remaining = max(0, FREE_TRIAL_SCANS - scans_used)
    return render_template(
        'dashboard.html',
        recent=[],
        total_scans=scans_used,
        total_defects=0,
        scans_remaining=scans_remaining,
        model_loaded=detector is not None
    )

@app.route('/workspace')
@login_required
def workspace():
    scans_used = user_scans.get(current_user.id, 0)
    scans_remaining = max(0, FREE_TRIAL_SCANS - scans_used)
    return render_template('workspace.html',
                           scans_remaining=scans_remaining,
                           model_loaded=detector is not None)

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    if detector is None:
        return jsonify({'error': 'Detection model is not loaded.'}), 503
    
    scans_used = user_scans.get(current_user.id, 0)
    if scans_used >= FREE_TRIAL_SCANS:
        return jsonify({'error': 'You have used all your free scans. Please upgrade.'}), 403
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read the uploaded image.'}), 400
    
    # Auto-rotate portrait images
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    results = detector.predict(image)
    
    # Draw boxes and save
    annotated = draw_detections(image.copy(), results)
    result_filename = f"result_{saved_filename}"
    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_filepath, annotated)
    
    # Update scan count
    user_scans[current_user.id] = scans_used + 1
    remaining = max(0, FREE_TRIAL_SCANS - (scans_used + 1))
    
    return jsonify({
        'success': True,
        'defect_count': results['defect_count'],
        'defect_detected': results['defect_detected'],
        'defects': results['defects'],
        'processing_time': results['processing_time'],
        'result_image_url': f"/uploads/{result_filename}",
        'scans_remaining': remaining
    })

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
    return render_template('history.html', records=[])

@app.route('/pricing')
def pricing():
    return render_template('landing.html', scroll_to='pricing')

@app.route('/demo')
def demo():
    flash('Create a free account to run your first scans.', 'info')
    return redirect(url_for('register'))

@app.route('/custom-request', methods=['POST'])
def custom_request():
    flash("Thanks - we'll follow up by email with a quote shortly.", 'success')
    return redirect(url_for('landing'))

@app.route('/upgrade/<plan>')
@login_required
def upgrade(plan):
    flash(f'Upgrade to {plan} plan - contact us for details.', 'info')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)