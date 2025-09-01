import React, { useState, useEffect } from 'react';
import Navbar from '../../components/Navbar';
import { useAuth, usePatientData, useHospital } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/PatientDashboard.module.css';

function PatientDashboard() {
  const { user, userProfile } = useAuth();
  const patientData = usePatientData();
  const hospitalData = useHospital();
  const [recentSessions, setRecentSessions] = useState([]);
  const [assignedDoctor, setAssignedDoctor] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedSessionId, setSelectedSessionId] = useState(null);

  useEffect(() => {
    if (userProfile) {
      fetchPatientData();
    }
  }, [userProfile]);

  const fetchPatientData = async () => {
    try {
      setIsLoading(true);

      // Fetch recent EEG sessions for this patient
      const { data: sessionsData, error: sessionsError } = await supabase
        .from('eeg_sessions')
        .select(`
          *,
          eeg_analysis_results(
            id,
            prediction,
            confidence_score,
            analysis_completed_at
          ),
          doctor_profiles!eeg_sessions_doctor_fkey(
            user_id,
            medical_license,
            specialization,
            user_profiles!doctor_profiles_user_fkey(
              full_name,
              email
            )
          )
        `)
        .eq('patient_id', userProfile.id)
        .order('created_at', { ascending: false })
        .limit(5);

      if (sessionsError) throw sessionsError;

      // Fetch assigned doctor information
      let doctorInfo = null;
      if (patientData?.assigned_doctor_id) {
        const { data: doctorData, error: doctorError } = await supabase
          .from('doctor_profiles')
          .select(`
            *,
            user_profiles!doctor_profiles_user_fkey(
              full_name,
              email,
              phone
            ),
            qualifications(qualification_name)
          `)
          .eq('user_id', patientData.assigned_doctor_id)
          .single();

        if (!doctorError) {
          doctorInfo = doctorData;
        }
      }

      setRecentSessions(sessionsData || []);
      setAssignedDoctor(doctorInfo);

    } catch (error) {
      // Error handled silently
    } finally {
      setIsLoading(false);
    }
  };

  const getVerificationStatus = () => {
    if (!patientData) return { status: 'unknown', message: 'Setting up your profile...' };
    
    switch (patientData.verification_status) {
      case 'pending':
        return { 
          status: 'pending', 
          message: 'Your profile is pending doctor verification',
          color: '#f59e0b'
        };
      case 'verified':
        return { 
          status: 'verified', 
          message: 'Your profile has been verified',
          color: '#10b981'
        };
      case 'rejected':
        return { 
          status: 'rejected', 
          message: 'Your profile verification was rejected',
          color: '#ef4444'
        };
      default:
        return { 
          status: 'unknown', 
          message: 'Verification status unknown',
          color: '#6b7280'
        };
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleSessionSelect = (sessionId) => {
    setSelectedSessionId(sessionId);
  };

  if (isLoading) {
    return (
      <>
        <Navbar />
        <div className={styles.loadingContainer}>
          <LoadingSpinner />
          <p>Loading your dashboard...</p>
        </div>
      </>
    );
  }

  const verificationStatus = getVerificationStatus();

  return (
    <>
      <Navbar />
      <div className={styles.dashboard}>
        <div className={styles.dashboardHeader}>
          <div className={styles.welcomeSection}>
            <h1>Welcome back, {userProfile?.full_name}</h1>
            <p className={styles.subtitle}>Patient Dashboard</p>
            <div className={styles.patientId}>
              ID: <span>{userProfile?.unique_identifier}</span>
            </div>
          </div>
          <div className={styles.hospitalInfo}>
            <h3>Hospital Information</h3>
            <p>{hospitalData?.name}</p>
            <p className={styles.hospitalAddress}>{hospitalData?.address}</p>
          </div>
        </div>

        <div className={styles.dashboardGrid}>
          {/* Status Card */}
          <div className={styles.statusCard}>
            <h3>Account Status</h3>
            <div className={styles.statusBadge} style={{ backgroundColor: `${verificationStatus.color}20`, color: verificationStatus.color }}>
              {verificationStatus.message}
            </div>
            {patientData?.verification_status === 'pending' && (
              <p className={styles.statusNote}>
                Your assigned doctor will review and verify your profile soon.
              </p>
            )}
          </div>

          {/* Assigned Doctor Card */}
          <div className={styles.doctorCard}>
            <h3>Assigned Doctor</h3>
            {assignedDoctor ? (
              <div className={styles.doctorInfo}>
                <div className={styles.doctorDetails}>
                  <h4>{assignedDoctor.user_profiles.full_name}</h4>
                  <p>{assignedDoctor.specialization || 'General Practice'}</p>
                  <p>{assignedDoctor.qualifications?.qualification_name}</p>
                  <p className={styles.license}>License: {assignedDoctor.medical_license}</p>
                </div>
                <div className={styles.contactInfo}>
                  <p>📧 {assignedDoctor.user_profiles.email}</p>
                  <p>📞 {assignedDoctor.user_profiles.phone}</p>
                </div>
              </div>
            ) : (
              <p className={styles.noDoctorMessage}>
                No doctor assigned yet. Please contact hospital administration.
              </p>
            )}
          </div>

          {/* Medical Information */}
          <div className={styles.medicalCard}>
            <h3>Medical Information</h3>
            <div className={styles.medicalGrid}>
              <div className={styles.medicalItem}>
                <span className={styles.medicalLabel}>Blood Group:</span>
                <span>{patientData?.blood_groups?.blood_type || 'Not specified'}</span>
              </div>
              <div className={styles.medicalItem}>
                <span className={styles.medicalLabel}>Emergency Contact:</span>
                <span>{patientData?.emergency_contact_name || 'Not specified'}</span>
              </div>
              <div className={styles.medicalItem}>
                <span className={styles.medicalLabel}>Emergency Phone:</span>
                <span>{patientData?.emergency_contact_phone || 'Not specified'}</span>
              </div>
            </div>
            {patientData?.medical_history && (
              <div className={styles.medicalHistory}>
                <h4>Medical History</h4>
                <p>{patientData.medical_history}</p>
              </div>
            )}
            {patientData?.current_medications && (
              <div className={styles.medications}>
                <h4>Current Medications</h4>
                <p>{patientData.current_medications}</p>
              </div>
            )}
            {patientData?.allergies && (
              <div className={styles.allergies}>
                <h4>Known Allergies</h4>
                <p>{patientData.allergies}</p>
              </div>
            )}
          </div>

          {/* Recent EEG Sessions */}
          <div className={styles.sessionsCard}>
            <h3>Recent EEG Sessions</h3>
            {recentSessions.length > 0 ? (
              <div className={styles.sessionsList}>
                {recentSessions.map((session) => (
                  <div
                    key={session.id}
                    className={`${styles.sessionItem} ${selectedSessionId === session.id ? styles.selectedSession : ''}`}
                    onClick={() => handleSessionSelect(session.id)}
                  >
                    <div className={styles.sessionHeader}>
                      <h4>{session.filename}</h4>
                      <span className={`${styles.statusBadge} ${styles[session.status]}`}>
                        {session.status}
                      </span>
                    </div>
                    <div className={styles.sessionDetails}>
                      <p>Session Date: {formatDate(session.session_date)}</p>
                      <p>Doctor: {session.doctor_profiles?.user_profiles?.full_name}</p>
                      {session.eeg_analysis_results?.[0] && (
                        <div className={styles.analysisResult}>
                          <p>
                            <strong>Result:</strong> {session.eeg_analysis_results[0].prediction}
                          </p>
                          <p>
                            <strong>Confidence:</strong> {(session.eeg_analysis_results[0].confidence_score * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.noSessionsMessage}>
                No EEG sessions found. Your doctor will upload and analyze your EEG data.
              </p>
            )}
          </div>

          {/* Quick Actions */}
          <div className={styles.actionsCard}>
            <h3>Quick Actions</h3>
            <div className={styles.actionButtons}>
              <button 
                className={styles.actionButton}
                onClick={() => window.location.href = '/patient/reports'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                  <path d="M3,3V21H21V3H3M5,19V5H19V19H5M7,12H9V17H7V12M11,7H13V17H11V7M15,10H17V17H15V10Z"/>
                </svg>
                View All Reports
              </button>
              <button 
                className={styles.actionButton}
                onClick={() => window.location.href = '/profile'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                  <path d="M12,2C13.1,2 14,2.9 14,4C14,5.1 13.1,6 12,6C10.9,6 10,5.1 10,4C10,2.9 10.9,2 12,2M21,9V7L19,5.5C18.8,5.7 18.6,5.9 18.4,6.1C18.1,6.3 17.8,6.5 17.6,6.7L19,8.2V10.6L17.6,12.1C17.8,12.3 18.1,12.5 18.4,12.7C18.6,12.9 18.8,13.1 19,13.3L21,11.8V9.8L21,9M15,12C16.1,12 17,12.9 17,14V22H15V14H9V22H7V14C7,12.9 7.9,12 9,12H15Z"/>
                </svg>
                Edit Profile
              </button>
              <button 
                className={styles.actionButton}
                onClick={() => window.location.href = '/patient/history'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                  <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
                </svg>
                Medical History
              </button>
              {patientData?.prescription_url && (
                <button 
                  className={styles.actionButton}
                  onClick={() => window.open(patientData.prescription_url, '_blank')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
                  </svg>
                  View Prescription
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default withAuth(PatientDashboard, ['patient']);