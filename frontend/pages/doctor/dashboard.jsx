import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import UnifiedSidebar from '../../components/UnifiedSidebar';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/DoctorDashboard.module.css';

function DoctorDashboard() {
  const router = useRouter();
  const { user, userProfile, hospitalData } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardStats, setDashboardStats] = useState({
    totalPatients: 0,
    pendingAssessments: 0,
    completedSessions: 0,
    todayAppointments: 0
  });

  // Sync activeTab with URL query parameter
  useEffect(() => {
    if (router.isReady) {
      const tabFromUrl = router.query.tab || 'overview';
      setActiveTab(tabFromUrl);
    }
  }, [router.isReady, router.query.tab]);

  const [eegSessions, setEegSessions] = useState([]);
  const [technicianReports, setTechnicianReports] = useState([]);
  const [myPatients, setMyPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);

  useEffect(() => {
    if (userProfile?.id && userProfile?.role === 'doctor') {
      console.log('👨‍⚕️ Doctor Profile loaded:', {
        id: userProfile.id,
        role: userProfile.role,
        name: userProfile.full_name,
        email: userProfile.email
      });
      fetchDashboardData();
    }
  }, [userProfile]);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
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
      let stats = {
        totalPatients: 0,
        pendingAssessments: 0,
        completedSessions: 0,
        todayAppointments: 0
      };

      try {
        // Query through patient_profiles joined with user_profiles
        const { data, count: totalPatients } = await supabase
          .from('patient_profiles')
          .select('user_id, user_profiles!patient_profiles_user_fkey!inner(account_status)', { count: 'exact' })
          .eq('assigned_doctor_id', userProfile.id)
          .eq('user_profiles.account_status', 'active');
        stats.totalPatients = totalPatients || 0;
        console.log('📊 Total patients count:', totalPatients);
      } catch (error) {
        console.error('Error fetching patient count:', error);
      }

      try {
        // Fetch pending assessments from predictions table
        const { count: pendingPredictions } = await supabase
          .from('predictions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .eq('status', 'Pending');
        stats.pendingAssessments = pendingPredictions || 0;
      } catch (error) {
        console.error('Error fetching pending sessions:', error);
      }

      try {
        // Fetch completed assessments from predictions table
        const { count: completedPredictions } = await supabase
          .from('predictions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .eq('status', 'Completed');
        stats.completedSessions = completedPredictions || 0;
      } catch (error) {
        console.error('Error fetching completed sessions:', error);
      }

      try {
        const today = new Date().toISOString().split('T')[0];
        // Fetch today's assessments from predictions table
        const { count: todayPredictions } = await supabase
          .from('predictions')
          .select('id', { count: 'exact', head: true })
          .eq('doctor_id', userProfile.id)
          .gte('created_at', today);
        stats.todayAppointments = todayPredictions || 0;
      } catch (error) {
        console.error('Error fetching today sessions:', error);
      }

      setDashboardStats(stats);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
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
      console.log('🔍 Fetching patients for doctor ID:', userProfile?.id);

      // First, try a simpler query to see all patients assigned to this doctor
      const { data: allData, error: allError } = await supabase
        .from('patient_profiles')
        .select('*')
        .eq('assigned_doctor_id', userProfile.id);

      console.log('🔍 All patient_profiles for this doctor:', allData);
      console.log('🔍 Error if any:', allError);

      // Now fetch with joins
      const { data, error } = await supabase
        .from('patient_profiles')
        .select(`
          *,
          user_profiles!patient_profiles_user_fkey(
            id,
            full_name,
            email,
            phone,
            date_of_birth,
            address,
            unique_identifier,
            created_at,
            account_status
          ),
          blood_groups(blood_type)
        `)
        .eq('assigned_doctor_id', userProfile.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('❌ Error fetching my patients:', error);
        throw error;
      }

      console.log('✅ Fetched patients with joins:', data);
      console.log('📊 Total patients:', data?.length || 0);

      // Filter for active accounts on the frontend if needed
      const activePatients = data?.filter(p =>
        p.user_profiles?.account_status === 'active'
      ) || [];

      console.log('✅ Active patients after filtering:', activePatients.length);

      setMyPatients(activePatients);
    } catch (error) {
      console.error('❌ Error fetching my patients:', error);
    }
  };

  const fetchEEGSessions = async () => {
    try {
      // Fetch from predictions table with RELAXED filtering (show null doctor_id too)
      const { data, error } = await supabase
        .from('predictions')
        .select(`
          id,
          filename,
          status,
          prediction,
          created_at,
          patient_id,
          patient_name,
          doctor_id,
          doctor_name,
          session_code,
          probabilities,
          technical_pdf_url
        `)
        .or(`doctor_id.eq.${userProfile.id},doctor_id.is.null`)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) throw error;
      console.log(`✅ Fetched ${data?.length || 0} EEG sessions (including unassigned)`);
      setEegSessions(data || []);
    } catch (error) {
      console.error('❌ Error fetching EEG sessions:', error);
    }
  };

  const fetchTechnicianReports = async () => {
    try {
      // Fetch from predictions table - RELAXED: show reports with doctor_id match OR null
      const { data, error } = await supabase
        .from('predictions')
        .select(`
          id,
          filename,
          status,
          prediction,
          created_at,
          patient_name,
          session_code,
          technical_pdf_url,
          probabilities,
          doctor_id
        `)
        .or(`doctor_id.eq.${userProfile.id},doctor_id.is.null`)
        .not('technical_pdf_url', 'is', null)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) throw error;
      console.log(`✅ Fetched ${data?.length || 0} technical reports (including unassigned)`);
      setTechnicianReports(data || []);
    } catch (error) {
      console.error('❌ Error fetching technical reports:', error);
    }
  };

  const fetchPatientDetails = async (patientId) => {
    try {
      setIsLoading(true);

      const { data, error } = await supabase
        .from('patient_profiles')
        .select(`
          *,
          user_profiles!patient_profiles_user_fkey!inner(*),
          blood_groups(*)
        `)
        .eq('user_id', patientId)
        .single();

      if (error) throw error;

      console.log('📋 Patient details loaded:', data);

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

  const navigationItems = [
    { id: 'overview', label: 'Dashboard', icon: 'Dashboard' },
    { id: 'patients', label: 'My Patients', icon: 'Users', badgeKey: 'totalPatients' },
    { id: 'eeg-sessions', label: 'EEG Sessions', icon: 'Activity', badgeKey: 'pendingAssessments' },
    { id: 'reports', label: 'Reports', icon: 'FileText' },
  ];

  const renderOverview = () => (
    <>
      <div className={styles.overviewGrid}>
        <div className={styles.statCard} onClick={() => setActiveTab('patients')}>
          <div className={styles.statIconWrapper}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
            </svg>
          </div>
          <div className={styles.statContent}>
            <h3>My Patients</h3>
            <div className={styles.statNumber}>{dashboardStats.totalPatients}</div>
            <p>Assigned to you</p>
          </div>
        </div>

        <div className={styles.statCard} onClick={() => setActiveTab('eeg-sessions')}>
          <div className={styles.statIconWrapper}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <div className={styles.statContent}>
            <h3>Pending Analysis</h3>
            <div className={styles.statNumber}>{dashboardStats.pendingAssessments}</div>
            <p>Awaiting review</p>
          </div>
        </div>

        <div className={styles.statCard}>
          <div className={styles.statIconWrapper}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <div className={styles.statContent}>
            <h3>Completed Sessions</h3>
            <div className={styles.statNumber}>{dashboardStats.completedSessions}</div>
            <p>Analysis complete</p>
          </div>
        </div>

        <div className={styles.statCard}>
          <div className={styles.statIconWrapper}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
              <line x1="16" y1="2" x2="16" y2="6"/>
              <line x1="8" y1="2" x2="8" y2="6"/>
              <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
          </div>
          <div className={styles.statContent}>
            <h3>Today's Sessions</h3>
            <div className={styles.statNumber}>{dashboardStats.todayAppointments}</div>
            <p>Uploaded today</p>
          </div>
        </div>
      </div>

      {/* Recent Activity Section */}
      <div className={styles.activitySection}>
        <div className={styles.activityColumn}>
          <div className={styles.sectionHeaderSmall}>
            <h3>Recent Patients</h3>
            <button onClick={() => setActiveTab('patients')} className={styles.viewAllBtn}>
              View All →
            </button>
          </div>
          <div className={styles.recentList}>
            {myPatients.slice(0, 5).map(patient => (
              <div key={patient.user_id} className={styles.recentItem}>
                <div className={styles.recentItemAvatar}>
                  {patient.user_profiles?.full_name?.charAt(0)?.toUpperCase() || 'P'}
                </div>
                <div className={styles.recentItemContent}>
                  <h4>{patient.user_profiles?.full_name}</h4>
                  <p>ID: {patient.user_profiles?.unique_identifier}</p>
                </div>
                <button
                  onClick={() => fetchPatientDetails(patient.user_id)}
                  className={styles.quickViewBtn}
                >
                  View
                </button>
              </div>
            ))}
            {myPatients.length === 0 && (
              <div className={styles.emptyStateSmall}>
                <p>No patients yet</p>
              </div>
            )}
          </div>
        </div>

        <div className={styles.activityColumn}>
          <div className={styles.sectionHeaderSmall}>
            <h3>Recent EEG Sessions</h3>
            <button onClick={() => setActiveTab('eeg-sessions')} className={styles.viewAllBtn}>
              View All →
            </button>
          </div>
          <div className={styles.recentList}>
            {eegSessions.slice(0, 5).map(session => (
              <div key={session.id} className={styles.recentItem}>
                <div className={styles.recentItemIcon}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                  </svg>
                </div>
                <div className={styles.recentItemContent}>
                  <h4>{session.session_code || `Session-${session.id.substring(0, 8)}`}</h4>
                  <p>{session.patient_name || 'Unknown Patient'}</p>
                </div>
                <span className={`${styles.statusBadgeSmall} ${styles[session.status?.toLowerCase()]}`}>
                  {session.status}
                </span>
              </div>
            ))}
            {eegSessions.length === 0 && (
              <div className={styles.emptyStateSmall}>
                <p>No sessions yet</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Latest Reports Preview */}
      {technicianReports.length > 0 && (
        <div className={styles.reportsPreview}>
          <div className={styles.sectionHeaderSmall}>
            <h3>Latest Technical Reports</h3>
            <button onClick={() => setActiveTab('reports')} className={styles.viewAllBtn}>
              View All →
            </button>
          </div>
          <div className={styles.reportsPreviewGrid}>
            {technicianReports.slice(0, 3).map(report => {
              const confidence = report.probabilities && Array.isArray(report.probabilities)
                ? (Math.max(...report.probabilities) * 100).toFixed(1)
                : 'N/A';

              return (
                <div key={report.id} className={styles.reportPreviewCard}>
                  <div className={styles.reportPreviewHeader}>
                    <h4>{report.session_code || `Session-${report.id.substring(0, 8)}`}</h4>
                    <span className={`${styles.statusBadgeSmall} ${styles[report.status?.toLowerCase()]}`}>
                      {report.status}
                    </span>
                  </div>
                  <div className={styles.reportPreviewDetails}>
                    <p><strong>Patient:</strong> {report.patient_name || 'Unknown'}</p>
                    <p><strong>Date:</strong> {new Date(report.created_at).toLocaleDateString()}</p>
                    {report.prediction && (
                      <p><strong>Result:</strong> <span style={{ color: report.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{report.prediction}</span></p>
                    )}
                    {confidence !== 'N/A' && (
                      <p><strong>Confidence:</strong> {confidence}%</p>
                    )}
                  </div>
                  {report.technical_pdf_url && (
                    <button
                      className={styles.downloadBtnSmall}
                      onClick={() => window.open(report.technical_pdf_url, '_blank')}
                    >
                      View Report
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );

  const renderMyPatients = () => (
    <div className={styles.patientManagement}>
      <div className={styles.sectionHeader}>
        <h2>My Patients</h2>
        <div className={styles.patientCount}>
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
              <p><strong>Emergency:</strong> {patient.emergency_contact_name || 'N/A'}</p>

              {patient.medical_history && (
                <div className={styles.medicalInfo}>
                  <p><strong>Medical History:</strong></p>
                  <p className={styles.textPreview}>{patient.medical_history}</p>
                </div>
              )}
            </div>

            <div className={styles.patientActions}>
              <button
                onClick={() => fetchPatientDetails(patient.user_id)}
                className={styles.viewDetailsBtn}
              >
                View Full Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {myPatients.length === 0 && (
        <div className={styles.emptyState}>
          <p>No patients assigned to you yet.</p>
        </div>
      )}
    </div>
  );

  const renderPatientDetails = () => {
    if (!patientDetails) return null;

    return (
      <div className={styles.patientDetailsView}>
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
              <span>Phone:</span>
              <span>{patientDetails.user_profiles?.phone}</span>
            </div>
          </div>

          <div className={styles.detailsCard}>
            <h3>Medical Information</h3>
            <div className={styles.detailRow}>
              <span>Blood Group:</span>
              <span>{patientDetails.blood_groups?.blood_type || 'Not specified'}</span>
            </div>

            {patientDetails.medical_history && (
              <div className={styles.medicalSection}>
                <h4>Medical History</h4>
                <p>{patientDetails.medical_history}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderEEGSessions = () => (
    <div className={styles.sessionsView}>
      <h2>EEG Sessions</h2>
      {eegSessions.length > 0 ? (
        <div className={styles.sessionsGrid}>
          {eegSessions.map(session => {
            const confidence = session.probabilities && Array.isArray(session.probabilities)
              ? (Math.max(...session.probabilities) * 100).toFixed(1)
              : 'N/A';

            return (
              <div key={session.id} className={styles.sessionCard}>
                <div className={styles.sessionHeader}>
                  <h3>{session.session_code || `Session-${session.id.substring(0, 8)}`}</h3>
                  <span className={`${styles.statusBadge} ${styles[session.status?.toLowerCase()]}`}>
                    {session.status}
                  </span>
                </div>
                <div className={styles.sessionInfo}>
                  <p><strong>Patient:</strong> {session.patient_name || 'Unknown'}</p>
                  <p><strong>File:</strong> {session.filename}</p>
                  <p><strong>Date:</strong> {new Date(session.created_at).toLocaleDateString()}</p>
                  {session.prediction && (
                    <p><strong>Result:</strong> <span style={{ color: session.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{session.prediction}</span></p>
                  )}
                  {confidence !== 'N/A' && (
                    <p><strong>Confidence:</strong> {confidence}%</p>
                  )}
                </div>
                {session.technical_pdf_url && (
                  <button
                    className={styles.downloadBtn}
                    onClick={() => window.open(session.technical_pdf_url, '_blank')}
                  >
                    View Technical Report
                  </button>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className={styles.emptyState}>
          <p>No EEG sessions found.</p>
        </div>
      )}
    </div>
  );

  const renderReports = () => (
    <div className={styles.reportsView}>
      <h2>Technical Reports</h2>
      {technicianReports.length > 0 ? (
        <div className={styles.reportsGrid}>
          {technicianReports.map(report => {
            const confidence = report.probabilities && Array.isArray(report.probabilities)
              ? (Math.max(...report.probabilities) * 100).toFixed(1)
              : 'N/A';

            return (
              <div key={report.id} className={styles.reportCard}>
                <div className={styles.reportHeader}>
                  <h3>{report.session_code || `Session-${report.id.substring(0, 8)}`}</h3>
                  <span className={`${styles.statusBadge} ${styles[report.status?.toLowerCase()]}`}>
                    {report.status}
                  </span>
                </div>
                <p><strong>Patient:</strong> {report.patient_name || 'Unknown'}</p>
                <p><strong>File:</strong> {report.filename}</p>
                <p><strong>Date:</strong> {new Date(report.created_at).toLocaleDateString()}</p>
                {report.prediction && (
                  <p><strong>Result:</strong> <span style={{ color: report.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{report.prediction}</span></p>
                )}
                {confidence !== 'N/A' && (
                  <p><strong>Confidence:</strong> {confidence}%</p>
                )}
                {report.technical_pdf_url && (
                  <button
                    className={styles.downloadBtn}
                    onClick={() => window.open(report.technical_pdf_url, '_blank')}
                  >
                    Download Technical Report
                  </button>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className={styles.emptyState}>
          <p>No reports available yet.</p>
        </div>
      )}
    </div>
  );

  // Handle tab change with URL update
  const handleTabChange = (tabId) => {
    router.push({
      pathname: router.pathname,
      query: { tab: tabId }
    }, undefined, { shallow: true });
  };

  if (isLoading) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.dashboardLayout}>
        <UnifiedSidebar
          user={user}
          userProfile={userProfile}
          hospitalData={hospitalData}
          activeTab={activeTab}
          onTabChange={handleTabChange}
          navigationItems={navigationItems}
          stats={dashboardStats}
        />

        <main className={styles.mainContent}>
          {selectedPatient ? (
            renderPatientDetails()
          ) : (
            <>
              {activeTab === 'overview' && renderOverview()}
              {activeTab === 'patients' && renderMyPatients()}
              {activeTab === 'eeg-sessions' && renderEEGSessions()}
              {activeTab === 'reports' && renderReports()}
            </>
          )}
        </main>
      </div>
    </>
  );
}

export default withAuth(DoctorDashboard, ['doctor']);
