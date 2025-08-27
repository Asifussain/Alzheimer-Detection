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
  
  // Patient management state
  const [myPatients, setMyPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);

  useEffect(() => {
    if (userProfile && hospitalData) {
      fetchDashboardData();
    }
  }, [userProfile, hospitalData]);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      await Promise.all([
        fetchDashboardStats(),
        fetchMyPatients()
      ]);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDashboardStats = async () => {
    try {
      // Get total patients assigned to this doctor
      const { count: totalPatients } = await supabase
        .from('patient_profiles')
        .select('user_id', { count: 'exact', head: true })
        .eq('assigned_doctor_id', userProfile.id);

      // Get pending assessments (patients without recent reports)
      // This would need to be implemented based on your assessment/report system
      const pendingAssessments = 0; // Placeholder

      // Get completed sessions (this would be based on your session tracking)
      const completedSessions = 0; // Placeholder

      // Get today's appointments (this would be based on your appointment system)
      const todayAppointments = 0; // Placeholder

      setDashboardStats({
        totalPatients: totalPatients || 0,
        pendingAssessments,
        completedSessions,
        todayAppointments
      });
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
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
        <div className={styles.statIcon}>👥</div>
        <div className={styles.statContent}>
          <h3>My Patients</h3>
          <div className={styles.statNumber}>{dashboardStats.totalPatients}</div>
          <p>Assigned to you</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>📋</div>
        <div className={styles.statContent}>
          <h3>Pending Assessments</h3>
          <div className={styles.statNumber}>{dashboardStats.pendingAssessments}</div>
          <p>Need attention</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>✅</div>
        <div className={styles.statContent}>
          <h3>Completed Sessions</h3>
          <div className={styles.statNumber}>{dashboardStats.completedSessions}</div>
          <p>This month</p>
        </div>
      </div>

      <div className={styles.statCard}>
        <div className={styles.statIcon}>📅</div>
        <div className={styles.statContent}>
          <h3>Today's Appointments</h3>
          <div className={styles.statNumber}>{dashboardStats.todayAppointments}</div>
          <p>Scheduled</p>
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
                👁️ View Full Details
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
          <button className={styles.secondaryBtn}>📈 View Assessment History</button>
          <button className={styles.secondaryBtn}>📝 Add Notes</button>
          <button className={styles.secondaryBtn}>📞 Schedule Appointment</button>
        </div>
      </div>
    );
  };

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
                className={activeTab === 'assessments' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('assessments')}
              >
                Assessments
              </button>
              <button 
                className={activeTab === 'reports' ? styles.activeTab : styles.tab}
                onClick={() => setActiveTab('reports')}
              >
                Reports
              </button>
            </div>

            <div className={styles.tabContent}>
              {activeTab === 'overview' && renderOverview()}
              {activeTab === 'patients' && renderMyPatients()}
              {activeTab === 'assessments' && (
                <div className={styles.comingSoon}>
                  <h2>Assessment Module</h2>
                  <p>This feature is coming soon! You'll be able to conduct cognitive assessments and track patient progress.</p>
                </div>
              )}
              {activeTab === 'reports' && (
                <div className={styles.comingSoon}>
                  <h2>Reports & Analytics</h2>
                  <p>This feature is coming soon! You'll be able to view detailed reports and analytics for your patients.</p>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}

export default withAuth(DoctorDashboard, ['doctor']);