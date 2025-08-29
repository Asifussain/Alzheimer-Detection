import os
import time
import hashlib
from functools import wraps
from flask import request, jsonify, g
from datetime import datetime, timedelta
from collections import defaultdict
import ipaddress

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)
failed_login_attempts = defaultdict(list)

# Security configuration
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 3600   # 1 hour in seconds
MAX_LOGIN_ATTEMPTS = 5     # max failed login attempts
LOCKOUT_DURATION = 900     # 15 minutes in seconds

# IP whitelist for admin endpoints (empty means no restrictions)
ADMIN_IP_WHITELIST = os.getenv('ADMIN_IP_WHITELIST', '').split(',') if os.getenv('ADMIN_IP_WHITELIST') else []

def get_client_ip():
    """Get the real client IP address, considering proxies"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr

def is_safe_ip(ip):
    """Check if IP is safe (not from suspicious ranges)"""
    try:
        ip_obj = ipaddress.ip_address(ip)
        
        # Block known suspicious ranges
        suspicious_ranges = [
            '10.0.0.0/8',      # Private network
            '172.16.0.0/12',   # Private network  
            '192.168.0.0/16',  # Private network
        ]
        
        # In production, you might want to be more selective
        # For development, we'll allow all IPs
        if os.getenv('FLASK_ENV') == 'development':
            return True
            
        for suspicious_range in suspicious_ranges:
            if ip_obj in ipaddress.ip_network(suspicious_range):
                return False
                
        return True
    except:
        return False

def rate_limit(max_requests=RATE_LIMIT_REQUESTS, window=RATE_LIMIT_WINDOW):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = get_client_ip()
            current_time = time.time()
            
            # Clean old requests
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip]
                if current_time - req_time < window
            ]
            
            # Check if rate limit exceeded
            if len(rate_limit_storage[client_ip]) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': window
                }), 429
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def check_failed_logins(email):
    """Check if account is locked due to failed login attempts"""
    current_time = time.time()
    
    # Clean old attempts
    failed_login_attempts[email] = [
        attempt_time for attempt_time in failed_login_attempts[email]
        if current_time - attempt_time < LOCKOUT_DURATION
    ]
    
    # Check if account is locked
    if len(failed_login_attempts[email]) >= MAX_LOGIN_ATTEMPTS:
        return True, LOCKOUT_DURATION - (current_time - failed_login_attempts[email][0])
    
    return False, 0

def record_failed_login(email):
    """Record a failed login attempt"""
    failed_login_attempts[email].append(time.time())

def clear_failed_logins(email):
    """Clear failed login attempts after successful login"""
    if email in failed_login_attempts:
        del failed_login_attempts[email]

def validate_input_length(data, field, min_length=1, max_length=255):
    """Validate input field length"""
    if field not in data:
        return f"{field} is required"
    
    value = data[field]
    if not isinstance(value, str):
        return f"{field} must be a string"
    
    if len(value) < min_length:
        return f"{field} must be at least {min_length} characters long"
    
    if len(value) > max_length:
        return f"{field} must be no more than {max_length} characters long"
    
    return None

def sanitize_input(data):
    """Sanitize input data to prevent XSS and other attacks"""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Basic sanitization - remove potential script tags
                value = value.replace('<script>', '').replace('</script>', '')
                value = value.replace('javascript:', '').replace('data:', '')
                value = value.strip()
            sanitized[key] = value
        return sanitized
    return data

def require_https(f):
    """Decorator to require HTTPS in production"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if os.getenv('FLASK_ENV') != 'development' and not request.is_secure:
            return jsonify({'error': 'HTTPS required'}), 400
        return f(*args, **kwargs)
    return decorated_function

def admin_ip_restriction(f):
    """Decorator to restrict admin endpoints to specific IPs"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not ADMIN_IP_WHITELIST:
            return f(*args, **kwargs)
        
        client_ip = get_client_ip()
        if client_ip not in ADMIN_IP_WHITELIST:
            return jsonify({'error': 'Access denied'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def log_security_event(event_type, details, user_id=None):
    """Log security events for monitoring"""
    timestamp = datetime.now().isoformat()
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    log_entry = {
        'timestamp': timestamp,
        'event_type': event_type,
        'details': details,
        'user_id': user_id,
        'ip_address': client_ip,
        'user_agent': user_agent
    }
    
    # In production, send this to a proper logging system
    if os.getenv('FLASK_ENV') == 'development':
        print(f"SECURITY EVENT: {log_entry}")
    
    # TODO: Integrate with your logging system (ELK stack, CloudWatch, etc.)

def validate_password_complexity(password):
    """Validate password meets complexity requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password must be no more than 128 characters long"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    # Check for common weak passwords
    weak_passwords = [
        'password', '12345678', 'password123', 'admin123',
        'qwerty123', 'letmein', 'welcome123', 'password1'
    ]
    
    if password.lower() in weak_passwords:
        return False, "Password is too common. Please choose a stronger password"
    
    return True, "Password meets complexity requirements"

def content_security_policy():
    """Add Content Security Policy headers"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['Content-Security-Policy'] = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "connect-src 'self';"
                )
                response.headers['X-Content-Type-Options'] = 'nosniff'
                response.headers['X-Frame-Options'] = 'DENY'
                response.headers['X-XSS-Protection'] = '1; mode=block'
            return response
        return decorated_function
    return decorator

# Input validation schemas
REGISTRATION_SCHEMA = {
    'email': {'required': True, 'min_length': 5, 'max_length': 255},
    'password': {'required': True, 'min_length': 8, 'max_length': 128},
    'full_name': {'required': True, 'min_length': 2, 'max_length': 100},
    'phone': {'required': True, 'min_length': 10, 'max_length': 20},
    'hospital_id': {'required': True, 'min_length': 1, 'max_length': 50},
    'role': {'required': True, 'min_length': 1, 'max_length': 20}
}

LOGIN_SCHEMA = {
    'email': {'required': True, 'min_length': 5, 'max_length': 255},
    'password': {'required': True, 'min_length': 1, 'max_length': 128}
}

def validate_schema(data, schema):
    """Validate data against a schema"""
    errors = []
    
    for field, rules in schema.items():
        if rules.get('required', False) and field not in data:
            errors.append(f"{field} is required")
            continue
        
        if field in data:
            value = data[field]
            
            if not isinstance(value, str):
                errors.append(f"{field} must be a string")
                continue
            
            min_length = rules.get('min_length', 0)
            max_length = rules.get('max_length', float('inf'))
            
            if len(value) < min_length:
                errors.append(f"{field} must be at least {min_length} characters long")
            
            if len(value) > max_length:
                errors.append(f"{field} must be no more than {max_length} characters long")
    
    return errors