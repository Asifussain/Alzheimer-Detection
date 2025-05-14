import { useRouter } from 'next/router';
import Navbar from '../components/Navbar';
import Link from 'next/link';
import React, { useState } from 'react'; // Import useState
import { useAuth } from '../components/AuthProvider'; // Import useAuth
import ReportViewer from '../components/ReportViewer'; // Import ReportViewer
import styles from '../styles/ResultPage.module.css';

export default function ResultPage() {
  const router = useRouter();
  // Ensure prediction uses the correct string comparison
  const { prediction: predictionQuery, filename, prediction_id } = router.query;
  // Standardize prediction string for comparison and display
  const prediction = predictionQuery === "Alzheimer's" ? "Alzheimer's" : predictionQuery === "Normal" ? "Normal" : null;

  const { profile } = useAuth(); // Get user profile for role
  const [showReport, setShowReport] = useState(false); // State to toggle report view

  // Use the standardized prediction string for class assignment
  const statusClass = prediction === "Alzheimer's" ? styles.alzheimer : styles.normal;

  const handleViewReportClick = () => {
      setShowReport(true);
  };

  return (
    <>
      <Navbar />
      <div className={styles.resultPageContainer}>
        <h1 className={styles.pageTitle}>Analysis Complete</h1>

        {/* Check if prediction and filename are valid */}
        {prediction && filename ? (
          <>
            <div className={styles.resultCard}>
              <p className={styles.filename}>File Analyzed: {filename}</p>
              <div className={`${styles.predictionStatus} ${statusClass}`}>
                {/* Display the standardized prediction string */}
                <span>{prediction}</span>
              </div>
              <p className={styles.disclaimer}>
                (This analysis is based on the provided EEG data using the ADFormer model. Consult a healthcare professional for a definitive diagnosis.)
              </p>
            </div>

            {/* --- Actions --- */}
            <div className={styles.actionsContainer}>
                {/* Analyse Another File Button */}
               <Link href="/" className={styles.actionButton}>
                 Analyse Another File
               </Link>

              {/* View Report Button (only if prediction_id exists and report not shown) */}
               {prediction_id && !showReport && (
                    <button onClick={handleViewReportClick} className={styles.actionButton}>
                        View Detailed Report
                    </button>
                )}

               {/* View History Button REMOVED */}

            </div>

            {/* --- Conditionally Render Report Viewer --- */}
            {showReport && prediction_id && (
                 <div style={{marginTop: '2rem'}}> {/* Add some space */}
                     <ReportViewer
                        predictionId={prediction_id}
                        userRole={profile?.role}
                     />
                 </div>
             )}

          </>
        ) : (
          <div className={styles.loadingErrorContainer}>
             <p>Loading result or result data not found...</p>
              <Link href="/" className={styles.secondaryButton} style={{marginTop: '1rem'}}>
                 Go Home
              </Link>
          </div>
        )}
      </div>
    </>
  );
}