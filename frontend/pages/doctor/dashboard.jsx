import { useState, useEffect } from 'react';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/DashboardLayout.module.css';

function DoctorDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardStats, setDashboardStats] = useState({
    totalPatients: 0,
    pendingAssessments: 0,
    completedSessions: 0,
    todayAppointments: 0
  });
  
  // EEG sessions state
  const [eegSessions, setEegSessions] = useState([]);
  const [technicianReports, setTechnicianReports] = useState([]);
  
  // Patient management state
  const [myPatients, setMyPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);

  useEffect(() => {
    if (userProfile?.id && userProfile?.role === 'doctor') {
      fetchDashboardData();
    }
  }, [userProfile]);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      // Use Promise.allSettled instead of Promise.all to ensure all requests complete
      await Promise.allSettled([
        fetchDashboardStats(),
        fetchMyPatients(),
        fetchEEGSessions(),
        fetchTechnicianReports()
      ]);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDashboardStats = async () => {
    try {
      // Initialize default stats
      let stats = {
        totalPatients: 0,
        pendingAssessments: 0,
        completedSessions: 0,
        todayAppointments: 0
      };

      // Get total patients assigned to this doctor
      try {
        const { count: totalPatients } = await supabase
          .from('patient_profiles')
          .select('user_id', { count: 'exact', head: true })
          .eq('assigned_doctor_id', userProfile.id)
          .eq('verification_status', 'verified');
        stats.totalPatients = totalPatients || 0;
      } catch (error) {
        console.error('Error fetching patient count:', error);
      }

      // Get pending EEG sessions for this doctor
      try {
        const { count: pendingEEGSessions } = await supabase
          .from('eeg_sessions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .eq('status', 'uploaded');
        stats.pendingAssessments = pendingEEGSessions || 0;
      } catch (error) {
        console.error('Error fetching pending sessions:', error);
      }

      // Get completed EEG sessions for this doctor
      try {
        const { count: completedEEGSessions } = await supabase
          .from('eeg_sessions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .eq('status', 'completed');
        stats.completedSessions = completedEEGSessions || 0;
      } catch (error) {
        console.error('Error fetching completed sessions:', error);
      }

      // Get today's EEG sessions
      try {
        const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
        const { count: todayEEGSessions } = await supabase
          .from('eeg_sessions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .gte('session_date', today);
        stats.todayAppointments = todayEEGSessions || 0;
      } catch (error) {
        console.error('Error fetching today sessions:', error);
      }

      setDashboardStats(stats);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      // Set default stats on error
      setDashboardStats({
        totalPatients: 0,
        pendingAssessments: 0,
        completedSessions: 0,
        todayAppointments: 0
      });
    }
  };

  const fetchMyPatients = async () => {
    try {
      const { data, error } = await supabase
        .from('patient_profiles')
        .select(`
          *,
          user_profiles!inner(
            id,
            full_name,
            email,
            phone,
            date_of_birth,
            address,
            unique_identifier,
            created_at
          ),
          blood_groups(blood_type)
        `)
        .eq('assigned_doctor_id', userProfile.id)
        .eq('user_profiles.account_status', 'active')
        .order('user_profiles.created_at', { ascending: false });

      if (error) throw error;
      setMyPatients(data || []);
    } catch (error) {
      console.error('Error fetching my patients:', error);
    }
  };

  const fetchEEGSessions = async () => {
    try {
      const { data, error } = await supabase
        .from('eeg_sessions')
        .select(`
          *,
          patient:user_profiles!patient_id(
            full_name,
            patient_profiles(patient_id, blood_groups(blood_type))
          ),
          eeg_analysis_results(
            id,
            prediction,
            confidence_score,
            analysis_completed_at
          )
        `)
        .eq('doctor_id', userProfile.id)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) throw error;
      setEegSessions(data || []);
    } catch (error) {
      console.error('Error fetching EEG sessions:', error);
    }
  };

  const fetchTechnicianReports = async () => {
    try {
      const { data, error } = await supabase
        .from('reports')
        .select(`
          *,
          eeg_sessions!session_id(
            session_code,
            filename,
            session_date,
            patient:user_profiles!patient_id(full_name)
          )
        `)
        .eq('report_type', 'technical')
        .eq('eeg_sessions.doctor_id', userProfile.id)
        .order('generated_at', { ascending: false })
        .limit(10);

      if (error) throw error;
      setTechnicianReports(data || []);
    } catch (error) {
      console.error('Error fetching technician reports:', error);
    }
  };

  const fetchPatientDetails = async (patientId) => {
    try {
      setIsLoading(true);
      
      const { data, error } = await supabase
        .from('patient_profiles')
        .select(`
          *,
          user_profiles!inner(*),
          blood_groups(*)
        `)
        .eq('user_id', patientId)
        .single();

      if (error) throw error;
      
      setPatientDetails(data);
      setSelectedPatient(patientId);
    } catch (error) {
      console.error('Error fetching patient details:', error);
      alert('Error loading patient details. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const calculateAge = (dateOfBirth) => {
    const today = new Date();
    const birthDate = new Date(dateOfBirth);
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    
    return age;
  };

  const renderOverview = () => (
    <div className={styles.overviewGrid}>
      <div className={styles.statCard}>
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>My Patients</h3>
          <div className={styles.statNumber}>{dashboardStats.totalPatients}</div>
          <p>Assigned to you</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Pending EEG Analysis</h3>
          <div className={styles.statNumber}>{dashboardStats.pendingAssessments}</div>
          <p>Awaiting analysis</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>✅</div>
        <div className={styles.statContent}>
          <h3>Completed EEG Sessions</h3>
          <div className={styles.statNumber}>{dashboardStats.completedSessions}</div>
          <p>Analysis complete</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M9,10V12H7V10H9M13,10V12H11V10H13M17,10V12H15V10H17M19,3A2,2 0 0,1 21,5V19A2,2 0 0,1 19,21H5C3.89,21 3,20.1 3,19V5A2,2 0 0,1 5,3H6V1H8V3H16V1H18V3H19M19,19V8H5V19H19M9,14V16H7V14H9M13,14V16H11V14H13M17,14V16H15V14H17Z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Today's EEG Sessions</h3>
          <div className={styles.statNumber}>{dashboardStats.todayAppointments}</div>
          <p>Uploaded today</p>
        </div>
      </div>
    </div>
  );

  const renderMyPatients = () => (
    <div className={styles.patientManagement}>
      <div className={styles.managementHeader}>
        <h2>My Patients</h2>
        <div className={styles.patientStats}>
          <span>Total: {myPatients.length}</span>
        </div>
      </div>
      
      <div className={styles.patientGrid}>
        {myPatients.map(patient => (
          <div key={patient.user_id} className={styles.patientCard}>
            <div className={styles.patientHeader}>
              <h3>{patient.user_profiles?.full_name}</h3>
              <span className={styles.patientId}>{patient.user_profiles?.unique_identifier}</span>
            </div>
            
            <div className={styles.patientDetails}>
              <p><strong>Age:</strong> {calculateAge(patient.user_profiles?.date_of_birth)} years</p>
              <p><strong>Phone:</strong> {patient.user_profiles?.phone}</p>
              <p><strong>Blood Group:</strong> {patient.blood_groups?.blood_type || 'N/A'}</p>
              <p><strong>Emergency Contact:</strong> {patient.emergency_contact_name || 'N/A'}</p>
              <p><strong>Emergency Phone:</strong> {patient.emergency_contact_phone || 'N/A'}</p>
              
              {patient.medical_history && (
                <div className={styles.medicalInfo}>
                  <p><strong>Medical History:</strong></p>
                  <p className={styles.textPreview}>{patient.medical_history}</p>
                </div>
              )}
              
              {patient.current_medications && (
                <div className={styles.medicalInfo}>
                  <p><strong>Current Medications:</strong></p>
                  <p className={styles.textPreview}>{patient.current_medications}</p>
                </div>
              )}
              
              {patient.allergies && (
                <div className={styles.medicalInfo}>
                  <p><strong>Allergies:</strong></p>
                  <p className={styles.textPreview}>{patient.allergies}</p>
                </div>
              )}
              
              {patient.prescription_url && (
                <div className={styles.prescriptionSection}>
                  <p><strong>Prescription:</strong></p>
                  <a 
                    href={patient.prescription_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className={styles.prescriptionLink}
                  >
                    📄 View Prescription
                  </a>
                </div>
              )}
            </div>
            
            <div className={styles.patientActions}>
              <button 
                onClick={() => fetchPatientDetails(patient.user_id)}
                className={styles.viewDetailsBtn}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                  <path d="M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z"/>
                </svg>
                View Full Details
              </button>
              <button 
                className={styles.startSessionBtn}
                onClick={() => alert('Assessment module coming soon!')}
              >
                🧠 Start Assessment
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {myPatients.length === 0 && (
        <div className={styles.emptyState}>
          <p>No patients assigned to you yet.</p>
          <p>Contact your hospital administrator if you believe this is an error.</p>
        </div>
      )}
    </div>
  );

  const renderPatientDetails = () => {
    if (!patientDetails) return null;

    return (
      <div className={styles.patientDetailsView}>
        <div className={styles.detailsHeader}>
          <button 
            onClick={() => {
              setSelectedPatient(null);
              setPatientDetails(null);
            }}
            className={styles.backBtn}
          >
            ← Back to Patients
          </button>
          <h2>{patientDetails.user_profiles?.full_name}</h2>
        </div>
        
        <div className={styles.detailsGrid}>
          <div className={styles.detailsCard}>
            <h3>Personal Information</h3>
            <div className={styles.detailRow}>
              <span>Full Name:</span>
              <span>{patientDetails.user_profiles?.full_name}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Patient ID:</span>
              <span>{patientDetails.user_profiles?.unique_identifier}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Age:</span>
              <span>{calculateAge(patientDetails.user_profiles?.date_of_birth)} years</span>
            </div>
            <div className={styles.detailRow}>
              <span>Date of Birth:</span>
              <span>{new Date(patientDetails.user_profiles?.date_of_birth).toLocaleDateString()}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Phone:</span>
              <span>{patientDetails.user_profiles?.phone}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Email:</span>
              <span>{patientDetails.user_profiles?.email}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Address:</span>
              <span>{patientDetails.user_profiles?.address}</span>
            </div>
          </div>

          <div className={styles.detailsCard}>
            <h3>Medical Information</h3>
            <div className={styles.detailRow}>
              <span>Blood Group:</span>
              <span>{patientDetails.blood_groups?.blood_type || 'Not specified'}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Verification Status:</span>
              <span className={`${styles.statusBadge} ${styles[patientDetails.verification_status]}`}>
                {patientDetails.verification_status}
              </span>
            </div>
            
            {patientDetails.medical_history && (
              <div className={styles.medicalSection}>
                <h4>Medical History</h4>
                <p>{patientDetails.medical_history}</p>
              </div>
            )}
            
            {patientDetails.current_medications && (
              <div className={styles.medicalSection}>
                <h4>Current Medications</h4>
                <p>{patientDetails.current_medications}</p>
              </div>
            )}
            
            {patientDetails.allergies && (
              <div className={styles.medicalSection}>
                <h4>Allergies</h4>
                <p>{patientDetails.allergies}</p>
              </div>
            )}
          </div>

          <div className={styles.detailsCard}>
            <h3>Emergency Contact</h3>
            <div className={styles.detailRow}>
              <span>Name:</span>
              <span>{patientDetails.emergency_contact_name || 'Not specified'}</span>
            </div>
            <div className={styles.detailRow}>
              <span>Phone:</span>
              <span>{patientDetails.emergency_contact_phone || 'Not specified'}</span>
            </div>
          </div>

          {patientDetails.prescription_url && (
            <div className={styles.detailsCard}>
              <h3>Prescription</h3>
              <p>Uploaded on: {new Date(patientDetails.prescription_uploaded_at).toLocaleDateString()}</p>
              <a 
                href={patientDetails.prescription_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className={styles.prescriptionBtn}
              >
                📄 View Prescription Document
              </a>
            </div>
          )}
        </div>
        
        <div className={styles.patientActionPanel}>
          <button className={styles.primaryBtn}>🧠 Start New Assessment</button>
          <button className={styles.secondaryBtn}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M3,3V21H21V3H3M5,19V5H19V19H5M7,12H9V17H7V12M11,7H13V17H11V7M15,10H17V17H15V10Z"/>
            </svg>
            View Assessment History
          </button>
          <button className={styles.secondaryBtn}>📝 Add Notes</button>
          <button className={styles.secondaryBtn}>📞 Schedule Appointment</button>
        </div>
      </div>
    );
  };

  const renderEEGSessions = () => (
    <div className={styles.eegSessionsView}>
      <div className={styles.sectionHeader}>
        <h2>EEG Sessions</h2>
        <p>EEG sessions uploaded by radiologists for your patients</p>
      </div>

      {eegSessions.length > 0 ? (
        <div className={styles.sessionsGrid}>
          {eegSessions.map(session => (
            <div key={session.id} className={styles.sessionCard}>
              <div className={styles.sessionHeader}>
                <h3>{session.session_code || session.id.slice(0, 8)}</h3>
                <span className={`${styles.statusBadge} ${styles[`status-${session.status}`]}`}>
                  {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                </span>
              </div>

              <div className={styles.sessionInfo}>
                <p><strong>Patient:</strong> {session.patient?.full_name}</p>
                <p><strong>Patient ID:</strong> {session.patient?.patient_profiles?.patient_id}</p>
                <p><strong>File:</strong> {session.filename}</p>
                <p><strong>Duration:</strong> {session.session_duration} minutes</p>
                <p><strong>Date:</strong> {new Date(session.session_date).toLocaleDateString()}</p>
                <p><strong>Analysis Type:</strong> {session.analysis_type}</p>
              </div>

              {session.eeg_analysis_results?.[0] && (
                <div className={styles.analysisResults}>
                  <h4>Analysis Results</h4>
                  <p><strong>Prediction:</strong> {session.eeg_analysis_results[0].prediction}</p>
                  <p><strong>Confidence:</strong> {(session.eeg_analysis_results[0].confidence_score * 100).toFixed(1)}%</p>
                  <p><strong>Completed:</strong> {new Date(session.eeg_analysis_results[0].analysis_completed_at).toLocaleDateString()}</p>
                </div>
              )}

              {session.session_notes && (
                <div className={styles.sessionNotes}>
                  <p><strong>Notes:</strong> {session.session_notes}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className={styles.emptyState}>
          <p>No EEG sessions found for your patients yet.</p>
          <p>Radiologists will upload EEG data for analysis, and you'll see the sessions here.</p>
        </div>
      )}
    </div>
  );

  const renderTechnicianReports = () => (
    <div className={styles.reportsView}>
      <div className={styles.sectionHeader}>
        <h2>Technical Reports</h2>
        <p>Technical analysis reports generated from EEG sessions</p>
      </div>

      {technicianReports.length > 0 ? (
        <div className={styles.reportsGrid}>
          {technicianReports.map(report => (
            <div key={report.id} className={styles.reportCard}>
              <div className={styles.reportHeader}>
                <h3>Technical Report</h3>
                <span className={styles.reportDate}>
                  {new Date(report.generated_at).toLocaleDateString()}
                </span>
              </div>

              <div className={styles.reportInfo}>
                <p><strong>Session:</strong> {report.eeg_sessions?.session_code}</p>
                <p><strong>Patient:</strong> {report.eeg_sessions?.patient?.full_name}</p>
                <p><strong>File:</strong> {report.eeg_sessions?.filename}</p>
                <p><strong>Session Date:</strong> {new Date(report.eeg_sessions?.session_date).toLocaleDateString()}</p>
                <p className={`${styles.accessStatus} ${report.is_accessible ? styles.accessible : styles.restricted}`}>
                  {report.is_accessible ? 'Accessible' : 'Access Restricted'}
                </p>
              </div>

              {report.is_accessible && (
                <div className={styles.reportActions}>
                  <button
                    className={styles.downloadBtn}
                    onClick={() => {
                      const link = document.createElement('a');
                      link.href = report.report_url;
                      link.download = `technical-report-${report.eeg_sessions?.session_code}.pdf`;
                      link.target = '_blank';
                      link.click();
                    }}
                  >
                    📄 Download Technical Report
                  </button>
                </div>
              )}

              {report.access_expires_at && (
                <div className={styles.expiryInfo}>
                  <p><strong>Access Expires:</strong> {new Date(report.access_expires_at).toLocaleDateString()}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className={styles.emptyState}>
          <p>No technical reports available yet.</p>
          <p>Reports will appear here once EEG analysis is completed.</p>
        </div>
      )}
    </div>
  );

  if (isLoading) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading doctor dashboard...</p>
      </div>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.dashboardContainer}>
        <div className={styles.dashboardHeader}>
          <div className={styles.welcomeSection}>
            <h1>Doctor Dashboard</h1>
            <p>Welcome, Dr. {userProfile?.full_name}</p>
            <p>{hospitalData?.name}</p>
            {userProfile?.doctor_profiles?.[0] && (
              <p className={styles.specialization}>
                {userProfile.doctor_profiles[0].specialization} • 
                {userProfile.doctor_profiles[0].experience_years} years experience
              </p>
            )}
          </div>
        </div>

        {selectedPatient ? (
          renderPatientDetails()
        ) : (
          <>
            <div className={styles.tabNavigation}>
              <button 
                className={activeTab === 'overview' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('overview')}
              >
                Overview
              </button>
              <button 
                className={activeTab === 'patients' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('patients')}
              >
                My Patients ({dashboardStats.totalPatients})
              </button>
              <button 
                className={activeTab === 'eeg-sessions' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('eeg-sessions')}
              >
                EEG Sessions ({eegSessions.length})
              </button>
              <button 
                className={activeTab === 'reports' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('reports')}
              >
                Technical Reports ({technicianReports.length})
              </button>
            </div>

            <div className={styles.tabContent}>
              {activeTab === 'overview' && renderOverview()}
              {activeTab === 'patients' && renderMyPatients()}
              {activeTab === 'eeg-sessions' && renderEEGSessions()}
              {activeTab === 'reports' && renderTechnicianReports()}
            </div>
          </>
        )}
      </div>
    </>
  );
}

export default withAuth(DoctorDashboard, ['doctor']);