import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../components/AuthProvider';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import supabase from '../lib/supabaseClient';
import styles from '../styles/Auth.module.css';

export default function LoginPage() {
  const { user, userProfile, isLoading, session } = useAuth();
  const router = useRouter();
  const [loginMethod, setLoginMethod] = useState('email'); // 'email' or 'google'
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showPasswordReset, setShowPasswordReset] = useState(false);

  // Redirect if already logged in
  useEffect(() => {
    if (!isLoading && user && userProfile) {
      // Redirect based on user role
      if (userProfile.role && userProfile.account_status === 'active') {
        router.replace(`/${userProfile.role}/dashboard`);
      } else if (userProfile.account_status === 'pending') {
        router.replace('/account-pending');
      } else if (!userProfile.phone_verified) {
        router.replace('/VerifyPhone');
      }
    }
  }, [user, userProfile, isLoading, router]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (error) setError('');
  };

  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');

    try {
      const { email, password } = formData;

      // Basic validation
      if (!email.trim() || !password.trim()) {
        throw new Error('Please enter both email and password');
      }

      // Attempt login with Supabase Auth
      const { data, error: signInError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password: password
      });

      if (signInError) {
        // Handle specific error cases
        if (signInError.message.includes('Invalid login credentials')) {
          throw new Error('Incorrect email or password. Please check your credentials.');
        } else if (signInError.message.includes('Email not confirmed')) {
          throw new Error('Your email address has not been verified. Please check your inbox.');
        } else if (signInError.message.includes('Too many requests')) {
          throw new Error('Too many login attempts. Please wait a few minutes and try again.');
        } else {
          throw new Error(signInError.message);
        }
      }

      // Login successful - AuthProvider will handle the redirect
      console.log('Login successful:', data.user?.email);

    } catch (error) {
      console.error('Login error:', error);
      setError(error.message || 'Login failed. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoogleLogin = async () => {
    setIsSubmitting(true);
    setError('');

    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/`
        }
      });

      if (error) {
        throw error;
      }

    } catch (error) {
      console.error('Google login error:', error);
      setError('Google login failed. Please try again or use email login.');
      setIsSubmitting(false);
    }
  };

  const handleForgotPassword = () => {
    setShowPasswordReset(true);
  };

  // Show loading spinner while checking authentication status
  if (isLoading) {
    return (
      <>
        <Navbar />
        <div className={styles.authContainer}>
          <div className={styles.loadingContainer}>
            <LoadingSpinner />
            <p>Checking authentication status...</p>
          </div>
        </div>
      </>
    );
  }

  // Show password reset form if requested
  if (showPasswordReset) {
    return (
      <>
        <Navbar />
        <div className={styles.authContainer}>
          <div className={styles.authCard}>
            <div className={styles.authHeader}>
              <h1 className={styles.authTitle}>Reset Password</h1>
              <p className={styles.authDescription}>
                Enter your email to receive a password reset link
              </p>
            </div>

            {error && (
              <div className={styles.errorMessage}>
                ❌ {error}
              </div>
            )}

            {success && (
              <div className={styles.successMessage}>
                ✅ {success}
              </div>
            )}

            <div className={styles.authForm}>
              <button
                type="button"
                onClick={() => setShowPasswordReset(false)}
                className={styles.linkButton}
              >
                ← Back to Login
              </button>
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.authContainer}>
        <div className={styles.authCard}>
          <div className={styles.authHeader}>
            <h1 className={styles.authTitle}>Welcome to AI4NEURO</h1>
            <p className={styles.authDescription}>
              Sign in to access your healthcare dashboard
            </p>
          </div>

          {/* Login Method Selection */}
          <div className={styles.loginMethodSelection}>
            <div className={styles.methodTabs}>
              <button
                type="button"
                className={loginMethod === 'email' ? styles.activeMethodTab : styles.methodTab}
                onClick={() => setLoginMethod('email')}
              >
                🏥 Hospital Login
              </button>
              <button
                type="button"
                className={loginMethod === 'google' ? styles.activeMethodTab : styles.methodTab}
                onClick={() => setLoginMethod('google')}
              >
                🌐 Google Login
              </button>
            </div>
          </div>

          {error && (
            <div className={styles.errorMessage}>
              ❌ {error}
            </div>
          )}

          {loginMethod === 'email' ? (
            /* Email/Password Login Form */
            <form onSubmit={handleEmailLogin} className={styles.authForm}>
              <div className={styles.formGroup}>
                <label htmlFor="email">Email Address</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className={styles.formInput}
                  placeholder="Enter your hospital email"
                  disabled={isSubmitting}
                  required
                  autoComplete="email"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="password">Password</label>
                <div className={styles.passwordInputContainer}>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className={styles.formInput}
                    placeholder="Enter your password"
                    disabled={isSubmitting}
                    required
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    className={styles.passwordToggle}
                    onClick={() => setShowPassword(!showPassword)}
                    disabled={isSubmitting}
                  >
                    {showPassword ? '👁️' : '👁️‍🗨️'}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className={styles.primaryButton}
              >
                {isSubmitting ? (
                  <>
                    <LoadingSpinner size={16} />
                    Signing In...
                  </>
                ) : (
                  'Sign In'
                )}
              </button>

              <div className={styles.helpLinks}>
                <button
                  type="button"
                  onClick={handleForgotPassword}
                  className={styles.linkButton}
                  disabled={isSubmitting}
                >
                  Forgot Password?
                </button>
              </div>
            </form>
          ) : (
            /* Google Login Section */
            <div className={styles.authForm}>
              <div className={styles.googleLoginSection}>
                <div className={styles.googleLoginInfo}>
                  <h3>🌐 Sign in with Google</h3>
                  <p>Use your Google account to access AI4NEURO</p>
                  <ul className={styles.googleBenefits}>
                    <li>✓ Quick and secure access</li>
                    <li>✓ No password to remember</li>
                    <li>✓ Seamless authentication</li>
                  </ul>
                </div>

                <button
                  type="button"
                  onClick={handleGoogleLogin}
                  disabled={isSubmitting}
                  className={styles.googleButton}
                >
                  {isSubmitting ? (
                    <>
                      <LoadingSpinner size={16} />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <svg width="20" height="20" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                      </svg>
                      Continue with Google
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          <div className={styles.authFooter}>
            <div className={styles.infoMessage}>
              <h4>🏥 For Hospital Staff</h4>
              <p>If you're a doctor, patient, or administrator:</p>
              <ul>
                <li>Use the <strong>Hospital Login</strong> with credentials provided by your administrator</li>
                <li>Contact your hospital IT department if you need account assistance</li>
                <li>First-time users will need to change their temporary password</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}