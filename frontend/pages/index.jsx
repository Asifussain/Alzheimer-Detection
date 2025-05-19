// frontend/pages/index.jsx
import { useEffect, useRef } from 'react'; // Removed useState as it's not used for fileName here
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { useAuth, PENDING_ROLE_SELECTION } from '../components/AuthProvider'; // Import PENDING_ROLE_SELECTION
import { useRouter } from 'next/router';
import styles from '../styles/IndexPage.module.css';
import pageStyles from '../styles/PageLayout.module.css'; // For overall page container
import { FiUploadCloud, FiCpu, FiShield, FiBarChart2, FiZap, FiLock } from 'react-icons/fi';
import Link from 'next/link';
import supabase from '../lib/supabaseClient';

export default function Home() {
  const { user, profile, loading: authLoading, session } = useAuth();
  const router = useRouter();
  const fileInputRef = useRef(null); // Still keep for disabled form aesthetic

  useEffect(() => {
    if (!authLoading && user && profile) {
      // If role needs selection, AuthProvider or withAuth will redirect to /select-role.
      // If role is confirmed and valid, redirect to the specific dashboard.
      if (profile.role && profile.role !== PENDING_ROLE_SELECTION && profile.role_confirmed) {
        const dashboardPath = `/${profile.role}/dashboard`;
        console.log(`IndexPage: Role confirmed ('${profile.role}'), redirecting to ${dashboardPath}`);
        router.replace(dashboardPath);
      }
      // If role is PENDING_ROLE_SELECTION or !role_confirmed, no redirect from here;
      // AuthProvider or withAuth (if trying to access a protected route) will handle it.
    }
  }, [user, profile, authLoading, router]);

  const handleLogin = async () => {
     try {
         const { error } = await supabase.auth.signInWithOAuth({
           provider: 'google',
           options: { redirectTo: `${window.location.origin}/` } // Ensure redirect after login
         });
         if (error) {
             console.error("Google Sign-In Error:", error.message);
             alert(`Login failed: ${error.message}`);
         }
     } catch (error) {
         console.error("Unexpected Login Error:", error);
         alert("An unexpected error occurred during login.");
     }
  };

  // Show loading page if auth is loading OR if user is logged in but profile/role check is pending (and will redirect)
  if (authLoading || (user && (!profile || profile.role === PENDING_ROLE_SELECTION || !profile.role_confirmed))) {
    return (
      <>
        <Navbar />
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 70px)', gap: '1rem', backgroundColor: 'var(--background-start)' }}>
          <LoadingSpinner />
          <p style={{ color: 'var(--text-secondary)' }}>
            {authLoading ? 'Loading authentication...' : 'Checking profile & redirecting...'}
          </p>
        </div>
      </>
    );
  }

  // Render Logged-Out Index Page (if not loading and no user/session)
  if (!user && !session) {
    return (
      <>
        <Navbar />
        <main>
            <section className={styles.hero}>
              <div className={styles.heroTop}>
                <div className={styles.heroContent}>
                  <h1 className={styles.mainTitle}>AI-Powered Alzheimer's Pattern Detection</h1>
                  <p className={styles.tagline}>
                    Utilize advanced EEG analysis with our ADFormer model for early insights. Secure, non-invasive, and powered by cutting-edge research.
                  </p>
                  <div className={styles.ctaContainer}>
                     <p className={styles.loginPrompt}>Get Started with Your Analysis</p>
                     <button onClick={handleLogin} className={styles.primaryLoginBtn}>
                        Login / Sign Up with Google
                     </button>
                     <p className={styles.uploadDisabledMessage}>
                        Login required to upload and analyse EEG files.
                     </p>
                     <div className={styles.uploadFormDisabled}>
                        <input
                            type="file"
                            ref={fileInputRef}
                            className={styles.hiddenInput}
                            disabled
                        />
                        <div className={styles.fileInputTrigger}>
                            Input EEG data (.npy)
                        </div>
                        <button type="button" className={styles.analyseBtnDisabled} disabled>
                            Analyse
                        </button>
                     </div>
                  </div>
                </div>
                <div className={styles.heroImageContainer}>
                  <img src="/images/brain.png" alt="AI Brain Analysis Concept" className={styles.heroImage} />
                </div>
              </div>
            </section>

            <section className={styles.featuresSection}>
                <h2 className={styles.sectionTitle}>How AI4NEURO Works</h2>
                <div className={styles.featuresGrid}>
                    <div className={styles.featureCard}> <FiUploadCloud className={styles.featureIcon} /> <h3 className={styles.featureTitle}>1. Secure Upload</h3> <p className={styles.featureDescription}> Log in and easily upload your anonymized EEG data in the standard .npy format. </p> </div>
                    <div className={styles.featureCard}> <FiCpu className={styles.featureIcon} /> <h3 className={styles.featureTitle}>2. AI Analysis</h3> <p className={styles.featureDescription}> Our fine-tuned ADFormer model processes the complex patterns within your EEG signals. </p> </div>
                    <div className={styles.featureCard}> <FiBarChart2 className={styles.featureIcon} /> <h3 className={styles.featureTitle}>3. Clear Insights</h3> <p className={styles.featureDescription}> Receive a clear pattern indication (Normal/Alzheimer's) and access detailed reports. </p> </div>
                </div>
            </section>
             <section className={styles.featuresSection} style={{backgroundColor: 'var(--background-start)'}}>
                <h2 className={styles.sectionTitle}>Key Features & Technology</h2>
                <div className={styles.featuresGrid}>
                    <div className={styles.featureCard}> <FiZap className={styles.featureIcon} /> <h3 className={styles.featureTitle}>Early Pattern Insight</h3> <p className={styles.featureDescription}> Aids in identifying potential Alzheimer's-related patterns sooner, facilitating timely medical consultation. </p> </div>
                    <div className={styles.featureCard}> <FiShield className={styles.featureIcon} /> <h3 className={styles.featureTitle}>Non-Invasive Method</h3> <p className={styles.featureDescription}> Analysis relies solely on standard EEG recordings, a safe and established procedure. </p> </div>
                    <div className={styles.featureCard}> <FiLock className={styles.featureIcon} /> <h3 className={styles.featureTitle}>Secure & Private</h3> <p className={styles.featureDescription}> Your data is handled with care using secure authentication and storage practices via Supabase. </p> </div>
                </div>
                 <p style={{textAlign: 'center', marginTop: '3rem', color: 'var(--text-secondary)'}}>
                    Powered by the ADFormer deep learning model for time-series analysis. <Link href="/about" className={pageStyles.link}>Learn More</Link>
                 </p>
            </section>
        </main>
      </>
    );
  }

  // Fallback, should ideally be covered by the conditions above
  // or user is already redirected to their dashboard.
  return (
    <>
      <Navbar />
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 70px)', backgroundColor: 'var(--background-start)' }}>
        <p style={{ color: 'var(--text-primary)' }}>Loading your experience...</p>
      </div>
    </>
  );
}