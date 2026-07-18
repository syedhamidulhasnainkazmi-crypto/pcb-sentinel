"""
PCB Sentinel - Flask web app
Converted from the original PyQt5 desktop GUI. Core detection logic
(detector.py) is unchanged; this file adds web routes, auth,
subscription tiers, and free-trial scan tracking.

Live camera mode is intentionally NOT included in this version -
image upload only. The GUI's live-camera code can be revisited later.
"""

import os
import json
import secrets
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, flash, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
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

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pcb_sentinel.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ============ MODEL PATH ============
# Put your trained weights file here and update the filename if different.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

# ============ STRIPE (add your keys later) ============
# import stripe
# stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_...')
PRICE_IDS = {
    'starter': 'price_XXXXXXXXXXXX',       # $49/mo - create in Stripe dashboard
    'professional': 'price_XXXXXXXXXXXX',  # $99/mo
    'enterprise': 'price_XXXXXXXXXXXX',    # $199/mo
}

# ============ EMAIL (plug in Resend/SendGrid later) ============
def send_verification_email(to_email, token):
    """Placeholder - wire this up to Resend/SendGrid/Gmail SMTP.
    For now it just prints the verification link to the console so you
    can test the flow locally before adding a real email provider."""
    verify_link = url_for('verify_email', token=token, _external=True)
    print(f"[EMAIL] Verification link for {to_email}: {verify_link}")

# ============ LOGIN MANAGER ============
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============ LOAD MODEL (once, at startup) ============
detector = None
if os.path.exists(MODEL_PATH):
    try:
        detector = YOLOv8Detector(MODEL_PATH)
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
else:
    print(f"[WARN] Model file not found at {MODEL_PATH} - detection routes will error until it's added")

FREE_TRIAL_SCANS = 5

# ============ DATABASE MODELS ============
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    company_name = db.Column(db.String(150), nullable=True)

    marketing_consent = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(100), nullable=True)

    scans_used = db.Column(db.Integer, default=0)          # free trial counter
    plan = db.Column(db.String(30), default='trial')       # trial / starter / professional / enterprise
    is_subscribed = db.Column(db.Boolean, default=False)
    subscription_expiry = db.Column(db.DateTime, nullable=True)
    monthly_scan_limit = db.Column(db.Integer, default=FREE_TRIAL_SCANS)
    scans_this_period = db.Column(db.Integer, default=0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def scans_remaining(self):
        if self.plan == 'trial':
            return max(0, FREE_TRIAL_SCANS - self.scans_used)
        if self.plan == 'enterprise':
            return None  # unlimited
        return max(0, self.monthly_scan_limit - self.scans_this_period)

    def can_scan(self):
        remaining = self.scans_remaining()
        return remaining is None or remaining > 0


class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(200))
    result_image_filename = db.Column(db.String(200))
    defect_count = db.Column(db.Integer, default=0)
    defect_detected = db.Column(db.Boolean, default=False)
    processing_time = db.Column(db.Float, default=0.0)
    defects_detail = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CustomRequest(db.Model):
    """Custom Training / API Integration inquiries - these are quote-based,
    not self-serve checkout, so we just capture the lead here."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    request_type = db.Column(db.String(30))  # 'custom_training' or 'api_integration'
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    company = db.Column(db.String(150))
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ============ PUBLIC ROUTES ============

@app.route('/')
def landing():
    """Public landing page - hero, problem/solution, pricing, about, demo CTA."""
    return render_template('landing.html')


@app.route('/demo')
def demo():
    """Lets a visitor try the detector without an account, using a capped
    number of anonymous demo scans tracked by session (simple version:
    just redirect to register with a note - swap in session-based demo
    logic later if you want true no-signup trial)."""
    flash('Create a free account to run your first scans - no credit card needed.', 'info')
    return redirect(url_for('register'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        company = request.form.get('company', '').strip()
        marketing_consent = 'marketing_consent' in request.form

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
            return render_template('register.html')

        token = secrets.token_urlsafe(32)
        new_user = User(
            username=username, email=email, company_name=company,
            marketing_consent=marketing_consent,
            verification_token=token, is_verified=False,
            plan='trial', monthly_scan_limit=FREE_TRIAL_SCANS
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        send_verification_email(email, token)
        flash('Account created! Check your email to verify your account, then log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/verify/<token>')
def verify_email(token):
    user = User.query.filter_by(verification_token=token).first()
    if user:
        user.is_verified = True
        user.verification_token = None
        db.session.commit()
        flash('Email verified! You can now log in.', 'success')
    else:
        flash('Invalid or expired verification link.', 'error')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))


@app.route('/pricing')
def pricing():
    return render_template('landing.html', scroll_to='pricing')


@app.route('/support')
def support():
    return render_template('support.html')


# ============ APP ROUTES (require login) ============

@app.route('/dashboard')
@login_required
def dashboard():
    recent = (DetectionHistory.query
              .filter_by(user_id=current_user.id)
              .order_by(DetectionHistory.created_at.desc())
              .limit(6).all())
    total_scans = DetectionHistory.query.filter_by(user_id=current_user.id).count()
    total_defects = db.session.query(db.func.sum(DetectionHistory.defect_count)) \
        .filter_by(user_id=current_user.id).scalar() or 0

    return render_template(
        'dashboard.html',
        recent=recent,
        total_scans=total_scans,
        total_defects=total_defects,
        scans_remaining=current_user.scans_remaining(),
        model_loaded=detector is not None
    )


@app.route('/workspace')
@login_required
def workspace():
    return render_template('workspace.html',
                            scans_remaining=current_user.scans_remaining(),
                            model_loaded=detector is not None)


@app.route('/detect', methods=['POST'])
@login_required
def detect():
    if detector is None:
        return jsonify({'error': 'Detection model is not loaded on the server yet.'}), 503

    if not current_user.can_scan():
        return jsonify({'error': 'You have used all your available scans. Please upgrade your plan.'}), 403

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

    # Auto-rotate portrait PCB photos, matching the desktop GUI's behavior
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    results = detector.predict(image)

    # Draw boxes and save an annotated copy
    annotated = draw_detections(image.copy(), results)
    result_filename = f"result_{saved_filename}"
    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_filepath, annotated)

    # Update usage counters
    if current_user.plan == 'trial':
        current_user.scans_used += 1
    elif current_user.plan != 'enterprise':
        current_user.scans_this_period += 1

    # Save history
    history = DetectionHistory(
        user_id=current_user.id,
        image_filename=saved_filename,
        result_image_filename=result_filename,
        defect_count=results['defect_count'],
        defect_detected=results['defect_detected'],
        processing_time=results['processing_time'],
        defects_detail=json.dumps(results['defects'])
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({
        'success': True,
        'defect_count': results['defect_count'],
        'defect_detected': results['defect_detected'],
        'defects': results['defects'],
        'processing_time': results['processing_time'],
        'result_image_url': f"/uploads/{result_filename}",
        'scans_remaining': current_user.scans_remaining()
    })


@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/history')
@login_required
def history():
    records = (DetectionHistory.query
               .filter_by(user_id=current_user.id)
               .order_by(DetectionHistory.created_at.desc())
               .all())
    return render_template('history.html', records=records)


# ============ SUBSCRIPTIONS (Stripe placeholders) ============

@app.route('/upgrade/<plan>')
@login_required
def upgrade(plan):
    """Placeholder checkout route. Once you add your Stripe secret key
    and real Price IDs above, uncomment the stripe.checkout.Session.create
    block below to redirect to a real Stripe Checkout page."""
    if plan not in PRICE_IDS:
        flash('Unknown plan.', 'error')
        return redirect(url_for('pricing'))

    flash(f'Stripe checkout not yet connected - add your API key in app.py to enable "{plan}" checkout.', 'info')
    return redirect(url_for('dashboard'))

    # --- Real Stripe integration (uncomment once keys are set) ---
    # checkout_session = stripe.checkout.Session.create(
    #     payment_method_types=['card'],
    #     line_items=[{'price': PRICE_IDS[plan], 'quantity': 1}],
    #     mode='subscription',
    #     success_url=url_for('upgrade_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
    #     cancel_url=url_for('pricing', _external=True),
    #     client_reference_id=str(current_user.id)
    # )
    # return redirect(checkout_session.url, code=303)


@app.route('/custom-request', methods=['POST'])
def custom_request():
    """Captures Custom Training / API Integration inquiries as leads
    rather than a fixed-price checkout, since these are quote-based."""
    req = CustomRequest(
        user_id=current_user.id if current_user.is_authenticated else None,
        request_type=request.form.get('type', 'custom_training'),
        name=request.form.get('name', ''),
        email=request.form.get('email', ''),
        company=request.form.get('company', ''),
        message=request.form.get('message', '')
    )
    db.session.add(req)
    db.session.commit()
    flash("Thanks - we'll follow up by email with a quote shortly.", 'success')
    return redirect(url_for('landing'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
