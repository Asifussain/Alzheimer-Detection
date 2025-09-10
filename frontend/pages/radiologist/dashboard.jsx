import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import Navbar from '../../components/Navbar';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/DashboardLayout.module.css';

function RadiologistDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  const router = useRouter();
  
  // State management
  const [isLoading, setIsLoading] = useState(true);
  const [doctors, setDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [recentSessions, setRecentSessions] = useState([]);
  const [stats, setStats] = useState({
    totalSessions: 0,
    pendingSessions: 0,
    completedSessions: 0,
    todaySessions: 0
  });
  const [error, setError] = useState('');

  // Load initial data
  useEffect(() => {
    if (user && userProfile?.role === 'radiologist') {
      loadDashboardData();
    }
  }, [user, userProfile]);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      setError('');

      // Check if userProfile has required data
      if (!userProfile?.hospital_id) {
        throw new Error('Hospital information not available');
      }

      // Load doctors from same hospital with improved error handling
      let doctorsData = [];
      try {
        const { data, error: doctorsError } = await supabase
          .from('user_profiles')
          .select(`
            id,
            full_name,
            email,
            doctor_profiles!inner(
              medical_license,
              specialization,
              experience_years,
              verification_status
            )
          `)
          .eq('hospital_id', userProfile.hospital_id)
          .eq('role', 'doctor')
          .eq('account_status', 'active')
          .eq('doctor_profiles.verification_status', 'verified')
          .order('full_name');

        if (doctorsError) {
          console.error('Error loading doctors:', doctorsError);
        } else {
          doctorsData = data || [];
        }
      } catch (error) {
        console.error('Error in doctors query:', error);
      }
      
      setDoctors(doctorsData);

      // Load recent EEG sessions statistics with improved error handling
      let sessions = [];
      try {
        const { data, error: sessionsError } = await supabase
          .from('eeg_sessions')
          .select('id, status, created_at, session_date, session_code')
          .eq('hospital_id', userProfile.hospital_id)
          .order('created_at', { ascending: false });

        if (sessionsError) {
          console.error('Error loading sessions:', sessionsError);
        } else {
          sessions = data || [];
        }
      } catch (error) {
        console.error('Error in sessions query:', error);
      }

      const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
      
      setStats({
        totalSessions: sessions.length,
        pendingSessions: sessions.filter(s => s.status === 'uploaded').length,
        completedSessions: sessions.filter(s => s.status === 'completed').length,
        todaySessions: sessions.filter(s => {
          if (!s.session_date) return false;
          return s.session_date.split('T')[0] === today;
        }).length
      });

      // Load recent sessions for display
      setRecentSessions(sessions.slice(0, 5));

    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setError(`Failed to load dashboard data: ${error.message}`);
      // Set default values on error
      setDoctors([]);
      setStats({
        totalSessions: 0,
        pendingSessions: 0,
        completedSessions: 0,
        todaySessions: 0
      });
      setRecentSessions([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDoctorSelect = async (doctor) => {
    try {
      setSelectedDoctor(doctor);
      setSelectedPatient(null);
      setPatients([]);

      // Load patients from the same hospital (not necessarily assigned to the doctor)
      // since radiologists work with all patients in the hospital
      const { data: patientsData, error: patientsError } = await supabase
        .from('user_profiles')
        .select(`
          id,
          full_name,
          email,
          date_of_birth,
          unique_identifier,
          patient_profiles!inner(
            patient_id,
            blood_group_id,
            verification_status,
            blood_groups(blood_type)
          )
        `)
        .eq('hospital_id', userProfile.hospital_id)
        .eq('role', 'patient')
        .eq('account_status', 'active')
        .eq('patient_profiles.verification_status', 'verified');

      if (patientsError) {
        console.error('Error loading patients:', patientsError);
        setError('Failed to load patients for selected doctor.');
      } else {
        setPatients(patientsData || []);
      }

    } catch (error) {
      console.error('Error loading patients:', error);
      setError('Failed to load patients for selected doctor.');
    }
  };

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
  };

  const handleCreateSession = () => {
    if (selectedDoctor && selectedPatient) {
      router.push(
        `/radiologist/create-session?doctor=${selectedDoctor.id}&patient=${selectedPatient.id}`
      );
    }
  };

  const handleViewSession = (sessionId) => {
    router.push(`/radiologist/session/${sessionId}`);
  };

  if (isLoading) {
    return (
      <>
        <Navbar />
        <div className={styles.dashboardContainer}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '50vh' 
          }}>
            <LoadingSpinner />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.dashboardContainer}>
        {/* Header */}
        <div className={styles.dashboardHeader}>
          <div className={styles.welcomeSection}>
            <h1>Radiologist Dashboard</h1>
            <p>Welcome back, {userProfile?.full_name}</p>
            <p className={styles.hospitalName}>{hospitalData?.name}</p>
          </div>
        </div>

        {error && (
          <div className={styles.errorAlert}>
            <span>{error}</span>
            <button onClick={() => setError('')}>×</button>
          </div>
        )}

        {/* Stats Overview */}
        <div className={styles.overviewGrid}>
          <div className={styles.statCard}>
            <div className={styles.statIcon}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3,3V21H21V3H3M5,19V5H19V19H5M7,12H9V17H7V12M11,7H13V17H11V7M15,10H17V17H15V10Z"/>
              </svg>
            </div>
            <div className={styles.statContent}>
              <h3>Total Sessions</h3>
              <div className={styles.statNumber}>{stats.totalSessions}</div>
              <p>All EEG sessions</p>
            </div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statIcon}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M16.2,16.2L11,13V7H12.5V12.2L17,14.9L16.2,16.2Z"/>
              </svg>
            </div>
            <div className={styles.statContent}>
              <h3>Pending Analysis</h3>
              <div className={styles.statNumber}>{stats.pendingSessions}</div>
              <p>Awaiting analysis</p>
            </div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statIcon}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"/>
              </svg>
            </div>
            <div className={styles.statContent}>
              <h3>Completed</h3>
              <div className={styles.statNumber}>{stats.completedSessions}</div>
              <p>Analysis complete</p>
            </div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statIcon}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19,3A2,2 0 0,1 21,5V19A2,2 0 0,1 19,21H5C3.89,21 3,20.1 3,19V5A2,2 0 0,1 5,3H6V1H8V3H16V1H18V3H19M19,19V8H5V19H19M9,10V12H7V10H9M13,10V12H11V10H13M17,10V12H15V10H17Z"/>
              </svg>
            </div>
            <div className={styles.statContent}>
              <h3>Today's Sessions</h3>
              <div className={styles.statNumber}>{stats.todaySessions}</div>
              <p>Sessions today</p>
            </div>
          </div>
        </div>

        <div className={styles.mainContent}>
          {/* Doctor & Patient Selection */}
          <div className={styles.contentSection}>
            <div className={styles.sectionHeader}>
              <h2>Create New EEG Session</h2>
              <p>Select a doctor and patient to create a new EEG analysis session</p>
            </div>

            <div className={styles.selectionGrid}>
              {/* Doctor Selection */}
              <div className={styles.selectionPanel}>
                <h3>Select Doctor</h3>
                <div className={styles.itemsList}>
                  {doctors.map(doctor => (
                    <div
                      key={doctor.id}
                      className={`${styles.listItem} ${
                        selectedDoctor?.id === doctor.id ? styles.selectedItem : ''
                      }`}
                      onClick={() => handleDoctorSelect(doctor)}
                    >
                      <div className={styles.itemInfo}>
                        <h4>{doctor.full_name}</h4>
                        <p>{doctor.doctor_profiles.specialization}</p>
                        <span className={styles.itemMeta}>
                          {doctor.doctor_profiles.experience_years} years experience
                        </span>
                      </div>
                    </div>
                  ))}
                  {doctors.length === 0 && (
                    <div className={styles.emptyState}>
                      <p>No verified doctors found in your hospital.</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Patient Selection */}
              <div className={styles.selectionPanel}>
                <h3>Select Patient</h3>
                <div className={styles.itemsList}>
                  {selectedDoctor ? (
                    patients.length > 0 ? (
                      patients.map(patient => (
                        <div
                          key={patient.id}
                          className={`${styles.listItem} ${
                            selectedPatient?.id === patient.id ? styles.selectedItem : ''
                          }`}
                          onClick={() => handlePatientSelect(patient)}
                        >
                          <div className={styles.itemInfo}>
                            <h4>{patient.full_name}</h4>
                            <p>ID: {patient.unique_identifier || patient.patient_profiles?.patient_id || 'N/A'}</p>
                            <span className={styles.itemMeta}>
                              Blood Type: {patient.patient_profiles?.blood_groups?.blood_type || 'N/A'}
                            </span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className={styles.emptyState}>
                        <p>No patients found for selected doctor.</p>
                      </div>
                    )
                  ) : (
                    <div className={styles.emptyState}>
                      <p>Select a doctor first to view patients.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {selectedDoctor && selectedPatient && (
              <div className={styles.actionSection}>
                <button 
                  className={styles.primaryButton}
                  onClick={handleCreateSession}
                >
                  Create EEG Session
                </button>
                <button 
                  className={styles.primaryButton}
                  onClick={() => router.push(`/radiologist/analysis?doctor=${selectedDoctor.id}&patient=${selectedPatient.id}`)}
                  style={{ marginLeft: '12px' }}
                >
                  Start EEG Analysis
                </button>
              </div>
            )}
          </div>

          {/* Recent Sessions */}
          <div className={styles.contentSection}>
            <div className={styles.sectionHeader}>
              <h2>Recent EEG Sessions</h2>
              <p>Latest EEG analysis sessions in your hospital</p>
            </div>

            <div className={styles.sessionsTable}>
              {recentSessions.length > 0 ? (
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Session Code</th>
                      <th>Date</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentSessions.map(session => (
                      <tr key={session.id}>
                        <td>{session.session_code || session.id.slice(0, 8)}</td>
                        <td>{new Date(session.session_date).toLocaleDateString()}</td>
                        <td>
                          <span className={`${styles.statusBadge} ${styles[`status-${session.status}`]}`}>
                            {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                          </span>
                        </td>
                        <td>
                          <button
                            className={styles.secondaryButton}
                            onClick={() => handleViewSession(session.id)}
                            style={{ marginRight: '8px' }}
                          >
                            View Details
                          </button>
                          {session.status === 'uploaded' && (
                            <button
                              className={styles.primaryButton}
                              onClick={() => router.push(`/radiologist/analysis?session=${session.id}`)}
                              style={{ fontSize: '12px', padding: '4px 8px' }}
                            >
                              Start EEG Analysis
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className={styles.emptyState}>
                  <p>No EEG sessions found.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default withAuth(RadiologistDashboard, ['radiologist']);