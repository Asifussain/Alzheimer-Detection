// frontend/pages/index.jsx
import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { useAuth } from '../components/AuthProvider';
import { useRouter } from 'next/router';
// --- IMPORT THE NEW CSS MODULE ---
import styles from '../styles/IndexPage.module.css';
import { FiUploadCloud, FiCpu, FiShield, FiBarChart2, FiZap, FiLock } from 'react-icons/fi';
import Link from 'next/link';
import supabase from '../lib/supabaseClient';

export default function Home() {
  const [fileName, setFileName] = useState('Input EEG data (.npy)');
  const { user, profile, loading: authLoading, session } = useAuth();
  const router = useRouter();
  const fileInputRef = useRef(null);

  // --- Redirect Logic (Keep as is) ---
  useEffect(() => {
    if (!authLoading && user && profile?.role) {
      const role = profile.role;
      if (role === 'patient') {
        router.replace('/patient/dashboard');
      } else if (role === 'technician') {
        router.replace('/technician/dashboard');
      } else if (role === 'clinician') {
        router.replace('/clinician/dashboard');
      }
    }
  }, [user, profile, authLoading, router]);

  // --- Login Handler (Keep as is) ---
  const handleLogin = async () => {
     try {
         const { error } = await supabase.auth.signInWithOAuth({
           provider: 'google',
           options: { /* Optional: Add redirect URL */ }
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

  // --- Render Logic ---
   if (authLoading || (user && profile?.role)) {
        return (
            <>
             <Navbar />
             <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 80px)', gap: '1rem' }}>
                <LoadingSpinner />
                <p style={{ color: 'var(--text-secondary)' }}>{user ? 'Redirecting to dashboard...' : 'Loading...'}</p>
             </div>
            </>
        );
   }

  // Render Logged-Out Index Page
  if (!user || !session) {
    // --- Use classes from IndexPage.module.css (imported as styles) ---
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
                            {fileName}
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
                    {/* Feature Cards using styles from IndexPage.module.css */}
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
                    Powered by the ADFormer deep learning model for time-series analysis. <Link href="/about">Learn More</Link>
                 </p>
            </section>
        </main>
      </>
    );
  }

  return <Navbar />;
}
