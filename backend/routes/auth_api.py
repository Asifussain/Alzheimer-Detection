import os
import jwt
import bcrypt
import re
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from supabase_client_setup import get_supabase_client
import secrets
from functools import wraps
from security_middleware import (
    rate_limit, require_https, sanitize_input, validate_schema,
    REGISTRATION_SCHEMA, LOGIN_SCHEMA, check_failed_logins,
    record_failed_login, clear_failed_logins, log_security_event,
    validate_password_complexity, content_security_policy
)

auth_bp = Blueprint('auth', __name__)

# JWT Configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Valid password"

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def generate_jwt_token(user_data):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user_data['id'],
        'email': user_data['email'],
        'role': user_data.get('role'),
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.current_user = payload
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/register', methods=['POST'])
@rate_limit(max_requests=5, window=3600)  # 5 registrations per hour
@require_https
@content_security_policy()
def register():
    """Register new user with email and password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Sanitize input
        data = sanitize_input(data)
        
        # Validate schema
        validation_errors = validate_schema(data, REGISTRATION_SCHEMA)
        if validation_errors:
            log_security_event('registration_validation_failed', {'errors': validation_errors})
            return jsonify({'error': '; '.join(validation_errors)}), 400
        
        # Extract and validate fields
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        phone = data.get('phone', '').strip()
        hospital_id = data.get('hospital_id', '').strip()
        role = data.get('role', '').strip()
        
        # Validate email format
        if not validate_email(email):
            log_security_event('registration_invalid_email', {'email': email})
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength using security middleware
        password_valid, password_message = validate_password_complexity(password)
        if not password_valid:
            log_security_event('registration_weak_password', {'email': email})
            return jsonify({'error': password_message}), 400
        
        # Validate role
        if role not in ['patient', 'doctor', 'admin']:
            return jsonify({'error': 'Invalid role'}), 400
        
        supabase = get_supabase_client()
        
        # Check if email already exists
        existing_user = supabase.table('user_profiles').select('id').eq('email', email).execute()
        if existing_user.data:
            return jsonify({'error': 'Email already registered'}), 409
        
        # Verify hospital exists
        hospital = supabase.table('hospitals').select('*').eq('id', hospital_id).execute()
        if not hospital.data:
            return jsonify({'error': 'Invalid hospital'}), 400
        
        hospital_data = hospital.data[0]
        
        # Generate unique identifier
        # Get the next sequence number for this hospital and role
        role_prefix = {'patient': 'PAT', 'doctor': 'DOC', 'admin': 'ADM'}
        prefix = role_prefix[role]
        
        # Get count of existing users with this role at this hospital
        existing_count = supabase.table('user_profiles')\
            .select('id')\
            .eq('hospital_id', hospital_id)\
            .eq('role', role)\
            .execute()
        
        sequence = len(existing_count.data) + 1
        unique_identifier = f"{hospital_data['hospital_code']}-{prefix}-{sequence:04d}"
        
        # Hash password
        password_hash = hash_password(password)
        
        # Generate email verification token
        verification_token = secrets.token_urlsafe(32)
        
        # Create user in Supabase auth (for consistency, though not required for custom auth)
        # This creates the base auth record that user_profiles references
        auth_user = supabase.auth.admin.create_user({
            "email": email,
            "email_confirm": False,
            "password": secrets.token_urlsafe(16)  # Random password, won't be used
        })
        
        if not auth_user.user:
            return jsonify({'error': 'Failed to create user account'}), 500
        
        # Create user profile
        user_profile_data = {
            'id': auth_user.user.id,
            'hospital_id': hospital_id,
            'unique_identifier': unique_identifier,
            'full_name': full_name,
            'email': email,
            'phone': phone,
            'role': role,
            'password_hash': password_hash,
            'auth_provider': 'email',
            'email_verified': False,
            'email_verification_token': verification_token,
            'account_status': 'active',  # Set to active immediately
            'phone_verified': True  # Skip phone verification
        }
        
        # Handle optional fields
        if 'date_of_birth' in data:
            user_profile_data['date_of_birth'] = data['date_of_birth']
        if 'address' in data:
            user_profile_data['address'] = data['address']
        
        profile_result = supabase.table('user_profiles').insert(user_profile_data).execute()
        
        if not profile_result.data:
            # Clean up auth user if profile creation fails
            supabase.auth.admin.delete_user(auth_user.user.id)
            return jsonify({'error': 'Failed to create user profile'}), 500
        
        # Generate JWT token
        user_data = profile_result.data[0]
        token = generate_jwt_token(user_data)
        
        # Remove sensitive data from response
        user_data.pop('password_hash', None)
        user_data.pop('email_verification_token', None)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user_data,
            'token': token,
            'requires_verification': True
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@auth_bp.route('/login', methods=['POST'])
@rate_limit(max_requests=10, window=3600)  # 10 login attempts per hour
@require_https
@content_security_policy()
def login():
    """Authenticate user with email and password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Sanitize input
        data = sanitize_input(data)
        
        # Validate schema
        validation_errors = validate_schema(data, LOGIN_SCHEMA)
        if validation_errors:
            log_security_event('login_validation_failed', {'errors': validation_errors})
            return jsonify({'error': '; '.join(validation_errors)}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not validate_email(email):
            log_security_event('login_invalid_email', {'email': email})
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check for account lockout due to failed attempts
        is_locked, remaining_time = check_failed_logins(email)
        if is_locked:
            log_security_event('login_attempt_locked_account', {'email': email, 'remaining_time': remaining_time})
            return jsonify({
                'error': f'Account temporarily locked due to too many failed attempts. Try again in {int(remaining_time)} seconds.',
                'locked_until': datetime.now() + timedelta(seconds=remaining_time)
            }), 429
        
        supabase = get_supabase_client()
        
        # Find user by email with auth_provider = 'email'
        user_result = supabase.table('user_profiles')\
            .select('*, hospitals(*)')\
            .eq('email', email)\
            .eq('auth_provider', 'email')\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        user = user_result.data[0]
        
        # Verify password
        if not user.get('password_hash'):
            return jsonify({'error': 'Account not set up for email authentication'}), 401
        
        if not verify_password(password, user['password_hash']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check if account is active
        if user.get('account_status') != 'active':
            status_message = {
                'pending': 'Your account is pending approval',
                'suspended': 'Your account has been suspended',
                'inactive': 'Your account is inactive'
            }.get(user.get('account_status', ''), 'Your account is not active')
            
            return jsonify({
                'error': status_message,
                'account_status': user.get('account_status')
            }), 403
        
        # Generate JWT token
        token = generate_jwt_token(user)
        
        # Remove sensitive data from response
        user_data = user.copy()
        user_data.pop('password_hash', None)
        user_data.pop('email_verification_token', None)
        user_data.pop('password_reset_token', None)
        
        return jsonify({
            'message': 'Login successful',
            'user': user_data,
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@auth_bp.route('/verify-email', methods=['POST'])
def verify_email():
    """Verify user email with token"""
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'error': 'Verification token is required'}), 400
        
        supabase = get_supabase_client()
        
        # Find user with this verification token
        user_result = supabase.table('user_profiles')\
            .select('*')\
            .eq('email_verification_token', token)\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'Invalid or expired verification token'}), 400
        
        user = user_result.data[0]
        
        # Update user as verified
        update_result = supabase.table('user_profiles')\
            .update({
                'email_verified': True,
                'email_verification_token': None,
                'updated_at': datetime.now().isoformat()
            })\
            .eq('id', user['id'])\
            .execute()
        
        if update_result.data:
            return jsonify({'message': 'Email verified successfully'}), 200
        else:
            return jsonify({'error': 'Verification failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Email verification failed: {str(e)}'}), 500

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email or not validate_email(email):
            return jsonify({'error': 'Valid email is required'}), 400
        
        supabase = get_supabase_client()
        
        # Find user
        user_result = supabase.table('user_profiles')\
            .select('*')\
            .eq('email', email)\
            .eq('auth_provider', 'email')\
            .execute()
        
        # Always return success to prevent email enumeration
        if not user_result.data:
            return jsonify({'message': 'If an account exists with this email, you will receive password reset instructions'}), 200
        
        user = user_result.data[0]
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=1)
        
        # Update user with reset token
        supabase.table('user_profiles')\
            .update({
                'password_reset_token': reset_token,
                'password_reset_expires': expires_at.isoformat()
            })\
            .eq('id', user['id'])\
            .execute()
        
        # TODO: Send email with reset link
        # For now, we'll just return the token (remove this in production)
        return jsonify({
            'message': 'If an account exists with this email, you will receive password reset instructions',
            'reset_token': reset_token  # Remove this in production
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Password reset request failed: {str(e)}'}), 500

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token"""
    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('new_password')
        
        if not token or not new_password:
            return jsonify({'error': 'Reset token and new password are required'}), 400
        
        # Validate new password
        password_valid, password_message = validate_password(new_password)
        if not password_valid:
            return jsonify({'error': password_message}), 400
        
        supabase = get_supabase_client()
        
        # Find user with valid reset token
        now = datetime.now().isoformat()
        user_result = supabase.table('user_profiles')\
            .select('*')\
            .eq('password_reset_token', token)\
            .gt('password_reset_expires', now)\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        user = user_result.data[0]
        
        # Hash new password
        password_hash = hash_password(new_password)
        
        # Update password and clear reset token
        update_result = supabase.table('user_profiles')\
            .update({
                'password_hash': password_hash,
                'password_reset_token': None,
                'password_reset_expires': None,
                'updated_at': datetime.now().isoformat()
            })\
            .eq('id', user['id'])\
            .execute()
        
        if update_result.data:
            return jsonify({'message': 'Password reset successfully'}), 200
        else:
            return jsonify({'error': 'Password reset failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Password reset failed: {str(e)}'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@require_auth
def change_password():
    """Change password for authenticated user"""
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current and new passwords are required'}), 400
        
        # Validate new password
        password_valid, password_message = validate_password(new_password)
        if not password_valid:
            return jsonify({'error': password_message}), 400
        
        supabase = get_supabase_client()
        user_id = request.current_user['user_id']
        
        # Get current user
        user_result = supabase.table('user_profiles')\
            .select('password_hash')\
            .eq('id', user_id)\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'User not found'}), 404
        
        user = user_result.data[0]
        
        # Verify current password
        if not verify_password(current_password, user['password_hash']):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Hash new password
        new_password_hash = hash_password(new_password)
        
        # Update password
        update_result = supabase.table('user_profiles')\
            .update({
                'password_hash': new_password_hash,
                'updated_at': datetime.now().isoformat()
            })\
            .eq('id', user_id)\
            .execute()
        
        if update_result.data:
            return jsonify({'message': 'Password changed successfully'}), 200
        else:
            return jsonify({'error': 'Password change failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Password change failed: {str(e)}'}), 500

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    """Verify JWT token and return user data"""
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get fresh user data
        supabase = get_supabase_client()
        user_result = supabase.table('user_profiles')\
            .select('*, hospitals(*)')\
            .eq('id', payload['user_id'])\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_result.data[0]
        
        # Remove sensitive data
        user_data.pop('password_hash', None)
        user_data.pop('email_verification_token', None)
        user_data.pop('password_reset_token', None)
        
        return jsonify({
            'valid': True,
            'user': user_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Token verification failed: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get current user's profile"""
    try:
        supabase = get_supabase_client()
        user_id = request.current_user['user_id']
        
        user_result = supabase.table('user_profiles')\
            .select('*, hospitals(*)')\
            .eq('id', user_id)\
            .execute()
        
        if not user_result.data:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_result.data[0]
        
        # Remove sensitive data
        user_data.pop('password_hash', None)
        user_data.pop('email_verification_token', None)
        user_data.pop('password_reset_token', None)
        
        return jsonify({'user': user_data}), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get profile: {str(e)}'}), 500