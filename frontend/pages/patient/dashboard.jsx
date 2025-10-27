import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Navbar from '../../components/Navbar';
import UnifiedSidebar from '../../components/UnifiedSidebar';
import { useAuth, usePatientData, useHospital } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/PatientDashboard.module.css';

function PatientDashboard() {
  const router = useRouter();
  const { user, userProfile } = useAuth();
  const patientData = usePatientData();
  const hospitalData = useHospital();
  const [activeTab, setActiveTab] = useState('overview');
  const [recentSessions, setRecentSessions] = useState([]);
  const [assignedDoctor, setAssignedDoctor] = useState(null);
  const [patientReports, setPatientReports] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedSessionId, setSelectedSessionId] = useState(null);

  // Sync activeTab with URL query parameter
  useEffect(() => {
    if (router.isReady) {
      const tabFromUrl = router.query.tab || 'overview';
      setActiveTab(tabFromUrl);
    }
  }, [router.isReady, router.query.tab]);

  useEffect(() => {
    if (userProfile) {
      fetchPatientData();
    }
  }, [userProfile]);

  const fetchPatientData = async () => {
    try {
      setIsLoading(true);

      // Fetch EEG sessions from predictions table - RELAXED: show patient_id match OR user_id match
      const { data: sessionsData, error: sessionsError } = await supabase
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
          patient_pdf_url,
          user_id
        `)
        .or(`patient_id.eq.${userProfile.id},user_id.eq.${userProfile.id}`)
        .order('created_at', { ascending: false })
        .limit(10);

      if (sessionsError) {
        console.error('❌ Error fetching sessions:', sessionsError);
      } else {
        console.log(`✅ Fetched ${sessionsData?.length || 0} patient sessions (including unassigned)`);
        setRecentSessions(sessionsData || []);
      }

      // Fetch assigned doctor information
      let doctorInfo = null;
      if (patientData?.assigned_doctor_id) {
        const { data: doctorData } = await supabase
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

        if (doctorData) doctorInfo = doctorData;
      }

      // Fetch patient reports (PDFs) from predictions table - RELAXED filtering
      const { data: reportsData, error: reportsError } = await supabase
        .from('predictions')
        .select(`
          id,
          filename,
          status,
          prediction,
          created_at,
          session_code,
          patient_pdf_url,
          probabilities,
          patient_id,
          user_id
        `)
        .or(`patient_id.eq.${userProfile.id},user_id.eq.${userProfile.id}`)
        .not('patient_pdf_url', 'is', null)
        .order('created_at', { ascending: false })
        .limit(10);

      if (reportsError) {
        console.error('❌ Error fetching reports:', reportsError);
      } else {
        console.log(`✅ Fetched ${reportsData?.length || 0} patient reports (including unassigned)`);
        setPatientReports(reportsData || []);
      }

      setAssignedDoctor(doctorInfo);
    } catch (error) {
      console.error('❌ Error fetching patient data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const navigationItems = [
    { id: 'overview', label: 'Dashboard', icon: 'Dashboard' },
    { id: 'doctor', label: 'My Doctor', icon: 'Stethoscope' },
    { id: 'sessions', label: 'EEG Sessions', icon: 'Activity', badgeKey: 'sessions' },
    { id: 'reports', label: 'Reports', icon: 'FileText', badgeKey: 'reports' },
  ];

  const stats = {
    sessions: recentSessions.length,
    reports: patientReports.length,
    verified: patientData?.verification_status === 'verified' ? 1 : 0
  };

  const getVerificationStatus = () => {
    if (!patientData) return { status: 'unknown', message: 'Setting up...', color: '#6b7280' };

    const statusMap = {
      pending: { status: 'pending', message: 'Pending Verification', color: '#f59e0b' },
      verified: { status: 'verified', message: 'Verified', color: '#10b981' },
      rejected: { status: 'rejected', message: 'Verification Rejected', color: '#ef4444' }
    };

    return statusMap[patientData.verification_status] || { status: 'unknown', message: 'Unknown', color: '#6b7280' };
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

  const renderOverview = () => {
    const verificationStatus = getVerificationStatus();

    return (
      <>
        {/* Compact Status Bar */}
        {patientData?.verification_status !== 'verified' && (
          <div className={styles.statusBanner} style={{ backgroundColor: `${verificationStatus.color}15`, borderColor: verificationStatus.color }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <div>
              <strong style={{ color: verificationStatus.color }}>{verificationStatus.message}</strong>
              <p>Your doctor will verify your profile soon.</p>
            </div>
          </div>
        )}

        {/* Quick Stats */}
        <div className={styles.quickStatsGrid}>
          <div className={styles.quickStatCard}>
            <div className={styles.quickStatIcon} style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            </div>
            <div>
              <h3>Total Sessions</h3>
              <p className={styles.quickStatNumber}>{recentSessions.length}</p>
            </div>
          </div>

          <div className={styles.quickStatCard}>
            <div className={styles.quickStatIcon} style={{ background: 'linear-gradient(135deg, #10b981, #059669)' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
            </div>
            <div>
              <h3>Available Reports</h3>
              <p className={styles.quickStatNumber}>{patientReports.length}</p>
            </div>
          </div>

          <div className={styles.quickStatCard}>
            <div className={styles.quickStatIcon} style={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
                <circle cx="20" cy="10" r="2"/>
              </svg>
            </div>
            <div>
              <h3>My Doctor</h3>
              <p className={styles.quickStatName}>{assignedDoctor?.user_profiles?.full_name || 'Not Assigned'}</p>
            </div>
          </div>
        </div>

        {/* Recent Reports with Notifications */}
        <div className={styles.recentReportsSection}>
          <div className={styles.sectionHeaderSmall}>
            <h3>Recent Reports</h3>
            <button onClick={() => handleTabChange('sessions')} className={styles.viewAllBtn}>
              View All →
            </button>
          </div>

          {patientReports.length > 0 ? (
            <div className={styles.reportsGrid}>
              {patientReports.slice(0, 3).map((report) => {
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
                      <p><strong>Date:</strong> {formatDate(report.created_at)}</p>
                      {report.prediction && (
                        <p><strong>Result:</strong> <span style={{ color: report.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{report.prediction}</span></p>
                      )}
                      {confidence !== 'N/A' && (
                        <p><strong>Confidence:</strong> {confidence}%</p>
                      )}
                    </div>
                    {report.patient_pdf_url && (
                      <button
                        className={styles.viewReportBtn}
                        onClick={() => window.open(report.patient_pdf_url, '_blank')}
                      >
                        View Report
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className={styles.emptyStateSmall}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              <p>No reports available yet</p>
              <span>Your reports will appear here once your doctor reviews your EEG sessions</span>
            </div>
          )}
        </div>

        {/* Doctor & Medical Info Row */}
        <div className={styles.bottomInfoRow}>
          <div className={styles.doctorInfoCard}>
            <div className={styles.sectionHeaderSmall}>
              <h3>Assigned Doctor</h3>
              <button onClick={() => handleTabChange('doctor')} className={styles.viewAllBtn}>
                View Details →
              </button>
            </div>
            {assignedDoctor ? (
              <div className={styles.doctorQuickInfo}>
                <div className={styles.doctorAvatarSmall}>
                  {assignedDoctor.user_profiles.full_name.charAt(0)}
                </div>
                <div className={styles.doctorDetailsSmall}>
                  <h4>{assignedDoctor.user_profiles.full_name}</h4>
                  <p>{assignedDoctor.specialization || 'General Practice'}</p>
                  <p className={styles.contactItem}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                    </svg>
                    {assignedDoctor.user_profiles.phone}
                  </p>
                </div>
              </div>
            ) : (
              <div className={styles.emptyStateSmall}>
                <p>No doctor assigned yet</p>
              </div>
            )}
          </div>

          <div className={styles.medicalInfoCard}>
            <h3>Medical Information</h3>
            <div className={styles.medicalInfoGrid}>
              <div className={styles.medicalInfoItem}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                  <path d="M12 6v6l4 2"/>
                </svg>
                <div>
                  <span>Blood Group</span>
                  <strong>{patientData?.blood_groups?.blood_type || 'Not specified'}</strong>
                </div>
              </div>
              <div className={styles.medicalInfoItem}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                  <circle cx="9" cy="7" r="4"/>
                </svg>
                <div>
                  <span>Emergency Contact</span>
                  <strong>{patientData?.emergency_contact_name || 'Not specified'}</strong>
                </div>
              </div>
              <div className={styles.medicalInfoItem}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                </svg>
                <div>
                  <span>Emergency Phone</span>
                  <strong>{patientData?.emergency_contact_phone || 'Not specified'}</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  };

  const renderDoctor = () => (
    <div className={styles.doctorSection}>
      <h2>Assigned Doctor</h2>
      {assignedDoctor ? (
        <div className={styles.doctorCard}>
          <div className={styles.doctorInfo}>
            <div className={styles.doctorAvatar}>
              {assignedDoctor.user_profiles.full_name.charAt(0)}
            </div>
            <div>
              <h3>{assignedDoctor.user_profiles.full_name}</h3>
              <p>{assignedDoctor.specialization || 'General Practice'}</p>
              <p className={styles.license}>License: {assignedDoctor.medical_license}</p>
            </div>
          </div>
          <div className={styles.contactInfo}>
            <p>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                <polyline points="22,6 12,13 2,6"/>
              </svg>
              {assignedDoctor.user_profiles.email}
            </p>
            <p>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
              </svg>
              {assignedDoctor.user_profiles.phone}
            </p>
          </div>
        </div>
      ) : (
        <div className={styles.emptyState}>
          <p>No doctor assigned yet.</p>
        </div>
      )}
    </div>
  );

  const renderSessions = () => (
    <div className={styles.sessionsSection}>
      <h2>Recent EEG Sessions</h2>
      {recentSessions.length > 0 ? (
        <div className={styles.sessionsList}>
          {recentSessions.map((session) => {
            const confidence = session.probabilities && Array.isArray(session.probabilities)
              ? (Math.max(...session.probabilities) * 100).toFixed(1)
              : 'N/A';

            return (
              <div key={session.id} className={styles.sessionCard}>
                <div className={styles.sessionHeader}>
                  <h4>{session.session_code || `Session-${session.id.substring(0, 8)}`}</h4>
                  <span className={`${styles.statusBadge} ${styles[session.status?.toLowerCase()]}`}>
                    {session.status}
                  </span>
                </div>
                <div className={styles.sessionDetails}>
                  <p><strong>File:</strong> {session.filename}</p>
                  <p><strong>Date:</strong> {formatDate(session.created_at)}</p>
                  <p><strong>Doctor:</strong> {session.doctor_name || 'Not assigned'}</p>
                  {session.prediction && (
                    <div className={styles.analysisResult}>
                      <p><strong>Result:</strong> <span style={{ color: session.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{session.prediction}</span></p>
                      {confidence !== 'N/A' && (
                        <p><strong>Confidence:</strong> {confidence}%</p>
                      )}
                    </div>
                  )}
                  {session.patient_pdf_url && (
                    <button
                      className={styles.downloadButton}
                      onClick={() => window.open(session.patient_pdf_url, '_blank')}
                    >
                      View My Report
                    </button>
                  )}
                </div>
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
    <div className={styles.reportsSection}>
      <h2>My EEG Reports</h2>
      {patientReports.length > 0 ? (
        <div className={styles.reportsList}>
          {patientReports.map((report) => {
            const confidence = report.probabilities && Array.isArray(report.probabilities)
              ? (Math.max(...report.probabilities) * 100).toFixed(1)
              : 'N/A';

            return (
              <div key={report.id} className={styles.reportCard}>
                <div className={styles.reportHeader}>
                  <h4>{report.session_code || `Session-${report.id.substring(0, 8)}`}</h4>
                  <span className={`${styles.statusBadge} ${styles[report.status?.toLowerCase()]}`}>
                    {report.status}
                  </span>
                </div>
                <p><strong>File:</strong> {report.filename}</p>
                <p><strong>Date:</strong> {formatDate(report.created_at)}</p>
                {report.prediction && (
                  <p><strong>Result:</strong> <span style={{ color: report.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981' }}>{report.prediction}</span></p>
                )}
                {confidence !== 'N/A' && (
                  <p><strong>Confidence:</strong> {confidence}%</p>
                )}
                {report.patient_pdf_url && (
                  <button
                    className={styles.downloadButton}
                    onClick={() => window.open(report.patient_pdf_url, '_blank')}
                  >
                    Download Report
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

  if (isLoading) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading dashboard...</p>
      </div>
    );
  }

  // Handle tab change with URL update
  const handleTabChange = (tabId) => {
    router.push({
      pathname: router.pathname,
      query: { tab: tabId }
    }, undefined, { shallow: true });
  };

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
          stats={stats}
        />

        <main className={styles.mainContent}>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'doctor' && renderDoctor()}
          {activeTab === 'sessions' && renderSessions()}
          {activeTab === 'reports' && renderReports()}
        </main>
      </div>
    </>
  );
}

export default withAuth(PatientDashboard, ['patient']);
