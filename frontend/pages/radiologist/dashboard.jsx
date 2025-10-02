import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/RadiologistDashboard.module.css';

// Custom Icons (SVG)
const Icons = {
  Dashboard: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="7" height="7" rx="1"/>
      <rect x="14" y="3" width="7" height="7" rx="1"/>
      <rect x="14" y="14" width="7" height="7" rx="1"/>
      <rect x="3" y="14" width="7" height="7" rx="1"/>
    </svg>
  ),
  Stethoscope: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
      <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
      <circle cx="20" cy="10" r="2"/>
    </svg>
  ),
  Activity: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  ),
  FileText: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>
  ),
  Upload: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  ),
  ChevronLeft: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  ),
  Menu: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="3" y1="12" x2="21" y2="12"/>
      <line x1="3" y1="6" x2="21" y2="6"/>
      <line x1="3" y1="18" x2="21" y2="18"/>
    </svg>
  ),
};

function RadiologistDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  const router = useRouter();
  const fileInputRef = useRef(null);

  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [allPredictions, setAllPredictions] = useState([]);

  // Pagination and filtering state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [predictionFilter, setPredictionFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState({ start: '', end: '' });

  const [dashboardStats, setDashboardStats] = useState({
    totalDoctors: 0,
    totalPatients: 0,
    totalSessions: 0,
    pendingSessions: 0,
    completedSessions: 0,
    todaySessions: 0,
  });

  const [doctors, setDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [recentSessions, setRecentSessions] = useState([]);

  // EEG Upload modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadData, setUploadData] = useState({
    file: null,
    channelIndex: 0 // 0-18 for 19 channels
  });
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const electrodeOptions = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
  ];

  const fetchDashboardData = useCallback(async () => {
    try {
      setIsLoading(true);
      const { data: { session } } = await supabase.auth.getSession();

      if (!session?.access_token) {
        setError('Authentication required');
        setIsLoading(false);
        return;
      }

      // Fetch ALL predictions
      const { data: allPreds } = await supabase
        .from('predictions')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(100);

      setAllPredictions(allPreds || []);
      setRecentSessions(allPreds?.slice(0, 5) || []);

      let doctorsData = [];

      // Fetch doctors from API
      const doctorsResponse = await fetch('/api/radiologist/get-doctors', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ hospital_id: userProfile.hospital_id })
      });

      if (doctorsResponse.ok) {
        const doctorsResult = await doctorsResponse.json();
        if (doctorsResult.success && doctorsResult.data) {
          doctorsData = doctorsResult.data;
          setDoctors(doctorsData);
        }
      }

      // Use predictions for statistics instead of eeg_sessions
      const sessionsData = allPreds || [];

      const today = new Date().toISOString().split('T')[0];
      setRecentSessions(sessionsData?.slice(0, 10) || []);

      setDashboardStats({
        totalDoctors: doctorsData.length || 0,
        totalPatients: 0, // Will be calculated when doctor is selected
        totalSessions: sessionsData?.length || 0,
        pendingSessions: sessionsData?.filter(s => s.status === 'uploaded').length || 0,
        completedSessions: sessionsData?.filter(s => s.status === 'completed' || s.status === 'reports_generated').length || 0,
        todaySessions: sessionsData?.filter(s => s.session_date?.split('T')[0] === today).length || 0,
      });

    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setError('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  }, [userProfile]);

  useEffect(() => {
    if (user && userProfile && userProfile.role === 'radiologist') {
      fetchDashboardData();
    }
  }, [user, userProfile, fetchDashboardData]);

  const handleDoctorSelect = async (doctor) => {
    setSelectedDoctor(doctor);
    setSelectedPatient(null);
    setSessions([]);
    setActiveTab('patients');

    try {
      const { data: { session } } = await supabase.auth.getSession();

      const response = await fetch('/api/radiologist/get-doctor-patients', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ doctor_id: doctor.id })
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          setPatients(result.data.patients || []);
          setDashboardStats(prev => ({ ...prev, totalPatients: result.data.patients?.length || 0 }));
        }
      }
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const handlePatientSelect = async (patient) => {
    setSelectedPatient(patient);
    setActiveTab('sessions');

    // Fetch patient's predictions from backend
    try {
      const { data: predictionsData } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', patient.id)
        .order('created_at', { ascending: false });

      setSessions(predictionsData || []);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const handleUploadEEG = () => {
    if (!selectedPatient || !selectedDoctor) {
      alert('Please select a doctor and patient first');
      return;
    }
    setShowUploadModal(true);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type (.npy files)
      if (!file.name.endsWith('.npy')) {
        alert('Please select a .npy file');
        return;
      }
      setUploadData({ ...uploadData, file });
    }
  };

  const handleSubmitUpload = async () => {
    if (!uploadData.file) {
      alert('Please select an EEG file');
      return;
    }

    try {
      setIsUploading(true);
      setUploadProgress(10);

      const { data: { session: authSession } } = await supabase.auth.getSession();
      const { data: { user } } = await supabase.auth.getUser();

      setUploadProgress(20);

      // Call backend Flask API for prediction
      const formData = new FormData();
      formData.append('file', uploadData.file);
      formData.append('user_id', user.id);
      formData.append('channel_index', uploadData.channelIndex.toString());

      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:5001';
      const predictResponse = await fetch(`${backendUrl}/api/predict`, {
        method: 'POST',
        body: formData
      });

      setUploadProgress(50);

      if (!predictResponse.ok) {
        const errorData = await predictResponse.json();
        throw new Error(errorData.error || 'Failed to start prediction');
      }

      const predictResult = await predictResponse.json();
      const predictionId = predictResult.prediction_id;

      setUploadProgress(100);

      alert(`EEG file uploaded successfully! Analysis has been started.\nPrediction ID: ${predictionId}`);
      setShowUploadModal(false);
      setUploadData({
        file: null,
        channelIndex: 0
      });

      // Refresh dashboard data
      fetchDashboardData();
      if (selectedPatient) {
        handlePatientSelect(selectedPatient);
      }

    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Filter and paginate predictions
  const getFilteredPredictions = () => {
    let filtered = [...allPredictions];

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(pred =>
        pred.filename?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        pred.id?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(pred => pred.status?.toLowerCase() === statusFilter.toLowerCase());
    }

    // Prediction result filter
    if (predictionFilter !== 'all') {
      if (predictionFilter === 'alzheimers') {
        filtered = filtered.filter(pred => pred.prediction?.toLowerCase().includes('alz'));
      } else if (predictionFilter === 'normal') {
        filtered = filtered.filter(pred => pred.prediction?.toLowerCase().includes('normal'));
      }
    }

    // Date filter
    if (dateFilter.start) {
      filtered = filtered.filter(pred => new Date(pred.created_at) >= new Date(dateFilter.start));
    }
    if (dateFilter.end) {
      filtered = filtered.filter(pred => new Date(pred.created_at) <= new Date(dateFilter.end + 'T23:59:59'));
    }

    return filtered;
  };

  const getPaginatedPredictions = () => {
    const filtered = getFilteredPredictions();
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filtered.slice(startIndex, endIndex);
  };

  const totalPages = Math.ceil(getFilteredPredictions().length / itemsPerPage);

  // Delete prediction
  const handleDeletePrediction = async (predictionId) => {
    if (!confirm('Are you sure you want to delete this prediction? This action cannot be undone.')) {
      return;
    }

    try {
      const { error } = await supabase
        .from('predictions')
        .delete()
        .eq('id', predictionId);

      if (error) throw error;

      alert('Prediction deleted successfully');
      fetchDashboardData();
    } catch (error) {
      console.error('Delete error:', error);
      alert(`Failed to delete prediction: ${error.message}`);
    }
  };

  // Reset filters
  const resetFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setPredictionFilter('all');
    setDateFilter({ start: '', end: '' });
    setCurrentPage(1);
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
      <div className={styles.dashboardContainer}>
        {/* Sidebar */}
        <aside className={`${styles.sidebar} ${sidebarCollapsed ? styles.collapsed : ''}`}>
          <div className={styles.sidebarHeader}>
            <div className={styles.hospitalInfo}>
              <h3>{hospitalData?.name || 'Hospital Dashboard'}</h3>
              <p>Radiologist Portal</p>
            </div>
            <button
              className={styles.toggleBtn}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            >
              {sidebarCollapsed ? <Icons.Menu /> : <Icons.ChevronLeft />}
            </button>
          </div>

          <nav className={styles.navigation}>
            <button
              className={`${styles.navItem} ${activeTab === 'overview' ? styles.active : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              <Icons.Dashboard />
              <span>Overview</span>
            </button>

            <button
              className={`${styles.navItem} ${activeTab === 'allPredictions' ? styles.active : ''}`}
              onClick={() => setActiveTab('allPredictions')}
            >
              <Icons.FileText />
              <span>All Predictions</span>
              <span className={styles.badge}>{allPredictions.length}</span>
            </button>

            <button
              className={`${styles.navItem} ${activeTab === 'doctors' ? styles.active : ''}`}
              onClick={() => setActiveTab('doctors')}
            >
              <Icons.Stethoscope />
              <span>Doctors</span>
            </button>

            <button
              className={`${styles.navItem} ${activeTab === 'patients' ? styles.active : ''}`}
              onClick={() => setActiveTab('patients')}
              disabled={!selectedDoctor}
            >
              <Icons.Activity />
              <span>Patients</span>
              {selectedDoctor && <span className={styles.badge}>{patients.length}</span>}
            </button>

            <button
              className={`${styles.navItem} ${activeTab === 'sessions' ? styles.active : ''}`}
              onClick={() => setActiveTab('sessions')}
              disabled={!selectedPatient}
            >
              <Icons.FileText />
              <span>EEG Sessions</span>
              {selectedPatient && <span className={styles.badge}>{sessions.length}</span>}
            </button>
          </nav>
        </aside>

        {/* Main Content */}
        <main className={`${styles.mainContent} ${sidebarCollapsed ? styles.expanded : ''}`}>
          {error && (
            <div className={styles.errorSection}>
              <p>{error}</p>
            </div>
          )}

          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Dashboard Overview</h1>

              <div className={styles.statsGrid}>
                <div className={styles.statCard} onClick={() => setActiveTab('doctors')}>
                  <Icons.Stethoscope />
                  <div>
                    <h3>Total Doctors</h3>
                    <div className={styles.statNumber}>{dashboardStats.totalDoctors}</div>
                  </div>
                </div>

                <div className={styles.statCard}>
                  <Icons.Activity />
                  <div>
                    <h3>Total Patients</h3>
                    <div className={styles.statNumber}>{dashboardStats.totalPatients}</div>
                  </div>
                </div>

                <div className={styles.statCard} onClick={() => setActiveTab('allPredictions')}>
                  <Icons.FileText />
                  <div>
                    <h3>Total Predictions</h3>
                    <div className={styles.statNumber}>{dashboardStats.totalSessions}</div>
                  </div>
                </div>

                <div className={styles.statCard}>
                  <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                  </svg>
                  <div>
                    <h3>Pending Analysis</h3>
                    <div className={styles.statNumber}>{dashboardStats.pendingSessions}</div>
                  </div>
                </div>

                <div className={styles.statCard}>
                  <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                  </svg>
                  <div>
                    <h3>Completed</h3>
                    <div className={styles.statNumber}>{dashboardStats.completedSessions}</div>
                  </div>
                </div>
              </div>

              {/* Recent Sessions */}
              <div className={styles.recentSections}>
                <h2>Recent Predictions</h2>
                {recentSessions.length === 0 ? (
                  <div className={styles.emptyState}>
                    <p>No predictions yet</p>
                  </div>
                ) : (
                  <div className={styles.sessionsList}>
                    {recentSessions.map(session => (
                      <div key={session.id} className={styles.sessionCard}>
                        <div className={styles.sessionHeader}>
                          <div>
                            <h4>{session.filename}</h4>
                            <p className={styles.predictionId}>ID: {session.id.substring(0, 8)}...</p>
                          </div>
                          <span className={`${styles.statusBadge} ${styles[session.status?.toLowerCase()]}`}>
                            {session.status}
                          </span>
                        </div>
                        <div className={styles.sessionInfo}>
                          <p><strong>Date:</strong> {new Date(session.created_at).toLocaleDateString()}</p>
                          {session.prediction && (
                            <p>
                              <strong>Result:</strong>{' '}
                              <span style={{color: session.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981'}}>
                                {session.prediction}
                              </span>
                            </p>
                          )}
                        </div>
                        {session.status === 'Completed' && (
                          <div className={styles.sessionActions}>
                            {session.patient_pdf_url && (
                              <a href={session.patient_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.viewBtn}>
                                Patient
                              </a>
                            )}
                            {session.clinician_pdf_url && (
                              <a href={session.clinician_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.viewBtn}>
                                Clinician
                              </a>
                            )}
                            {session.technical_pdf_url && (
                              <a href={session.technical_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.reportBtn}>
                                Technical
                              </a>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* All Predictions Tab */}
          {activeTab === 'allPredictions' && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h1 className={styles.pageTitle}>All Predictions History</h1>
                <button className={styles.refreshBtn} onClick={fetchDashboardData}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
                  </svg>
                  Refresh
                </button>
              </div>

              {/* Filters Section */}
              <div className={styles.filtersSection}>
                <div className={styles.filterRow}>
                  <div className={styles.searchBox}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8"/>
                      <path d="m21 21-4.35-4.35"/>
                    </svg>
                    <input
                      type="text"
                      placeholder="Search by filename or ID..."
                      value={searchQuery}
                      onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                      className={styles.searchInput}
                    />
                  </div>

                  <select
                    value={statusFilter}
                    onChange={(e) => { setStatusFilter(e.target.value); setCurrentPage(1); }}
                    className={styles.filterSelect}
                  >
                    <option value="all">All Status</option>
                    <option value="pending">Pending</option>
                    <option value="processing">Processing</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                  </select>

                  <select
                    value={predictionFilter}
                    onChange={(e) => { setPredictionFilter(e.target.value); setCurrentPage(1); }}
                    className={styles.filterSelect}
                  >
                    <option value="all">All Results</option>
                    <option value="alzheimers">Alzheimer's</option>
                    <option value="normal">Normal</option>
                  </select>

                  <input
                    type="date"
                    value={dateFilter.start}
                    onChange={(e) => { setDateFilter({...dateFilter, start: e.target.value}); setCurrentPage(1); }}
                    className={styles.dateInput}
                    placeholder="Start Date"
                  />

                  <input
                    type="date"
                    value={dateFilter.end}
                    onChange={(e) => { setDateFilter({...dateFilter, end: e.target.value}); setCurrentPage(1); }}
                    className={styles.dateInput}
                    placeholder="End Date"
                  />

                  <button className={styles.clearFiltersBtn} onClick={resetFilters}>
                    Clear Filters
                  </button>
                </div>

                <div className={styles.resultsInfo}>
                  Showing {getPaginatedPredictions().length} of {getFilteredPredictions().length} predictions
                </div>
              </div>

              {getFilteredPredictions().length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.FileText />
                  <p>No predictions found</p>
                </div>
              ) : (
                <>
                  <div className={styles.predictionsTable}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Filename</th>
                          <th>Date</th>
                          <th>Status</th>
                          <th>Prediction</th>
                          <th>Confidence</th>
                          <th>Reports</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {getPaginatedPredictions().map(pred => (
                          <tr key={pred.id}>
                            <td>
                              <div className={styles.filenameCell}>
                                <strong>{pred.filename}</strong>
                                <small>ID: {pred.id.substring(0, 8)}...</small>
                              </div>
                            </td>
                            <td>{new Date(pred.created_at).toLocaleDateString()}</td>
                            <td>
                              <span className={`${styles.statusBadge} ${styles[pred.status?.toLowerCase()]}`}>
                                {pred.status}
                              </span>
                            </td>
                            <td>
                              {pred.prediction ? (
                                <strong style={{color: pred.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981'}}>
                                  {pred.prediction}
                                </strong>
                              ) : (
                                <span className={styles.mutedText}>-</span>
                              )}
                            </td>
                            <td>
                              {pred.probabilities && pred.probabilities.length > 0 ? (
                                <strong>{(Math.max(...pred.probabilities) * 100).toFixed(1)}%</strong>
                              ) : (
                                <span className={styles.mutedText}>-</span>
                              )}
                            </td>
                            <td>
                              {pred.status === 'Completed' ? (
                                <div className={styles.reportLinks}>
                                  {pred.patient_pdf_url && (
                                    <a href={pred.patient_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
                                      P
                                    </a>
                                  )}
                                  {pred.clinician_pdf_url && (
                                    <a href={pred.clinician_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
                                      C
                                    </a>
                                  )}
                                  {pred.technical_pdf_url && (
                                    <a href={pred.technical_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
                                      T
                                    </a>
                                  )}
                                </div>
                              ) : (
                                <span className={styles.mutedText}>-</span>
                              )}
                            </td>
                            <td>
                              <button
                                className={styles.deleteBtn}
                                onClick={() => handleDeletePrediction(pred.id)}
                                title="Delete prediction"
                              >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                </svg>
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className={styles.pagination}>
                      <button
                        className={styles.pageBtn}
                        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                        disabled={currentPage === 1}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="15 18 9 12 15 6"/>
                        </svg>
                        Previous
                      </button>

                      <div className={styles.pageNumbers}>
                        {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                          <button
                            key={page}
                            className={`${styles.pageNumber} ${currentPage === page ? styles.active : ''}`}
                            onClick={() => setCurrentPage(page)}
                          >
                            {page}
                          </button>
                        ))}
                      </div>

                      <button
                        className={styles.pageBtn}
                        onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                        disabled={currentPage === totalPages}
                      >
                        Next
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="9 18 15 12 9 6"/>
                        </svg>
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Doctors Tab */}
          {activeTab === 'doctors' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Select Doctor</h1>
              {doctors.length === 0 ? (
                <div className={styles.emptyState}>
                  <p>No doctors found in your hospital</p>
                </div>
              ) : (
                <div className={styles.cardGrid}>
                  {doctors.map(doctor => (
                    <div
                      key={doctor.id}
                      className={`${styles.doctorCard} ${selectedDoctor?.id === doctor.id ? styles.selected : ''}`}
                      onClick={() => handleDoctorSelect(doctor)}
                    >
                      <div className={styles.avatar}>
                        {doctor.full_name?.charAt(0)?.toUpperCase() || 'D'}
                      </div>
                      <h3>{doctor.full_name}</h3>
                      <p className={styles.specialization}>{doctor.specialization || 'General Practice'}</p>
                      <p className={styles.email}>{doctor.email}</p>
                      <div className={styles.doctorStats}>
                        <span>{doctor.experience_years || 0} years exp.</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Patients Tab */}
          {activeTab === 'patients' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>
                Patients of Dr. {selectedDoctor?.full_name}
              </h1>
              {patients.length === 0 ? (
                <div className={styles.emptyState}>
                  <p>No patients assigned to this doctor</p>
                </div>
              ) : (
                <div className={styles.cardGrid}>
                  {patients.map(patient => (
                    <div
                      key={patient.id}
                      className={`${styles.patientCard} ${selectedPatient?.id === patient.id ? styles.selected : ''}`}
                      onClick={() => handlePatientSelect(patient)}
                    >
                      <div className={styles.cardHeader}>
                        <h3>{patient.full_name}</h3>
                        <span className={styles.patientId}>{patient.unique_identifier}</span>
                      </div>
                      <div className={styles.cardBody}>
                        <p><strong>Email:</strong> {patient.email}</p>
                        <p><strong>DOB:</strong> {patient.date_of_birth ? new Date(patient.date_of_birth).toLocaleDateString() : 'N/A'}</p>
                        {patient.patient_profiles?.[0]?.blood_groups && (
                          <p><strong>Blood Group:</strong> {patient.patient_profiles[0].blood_groups.blood_type}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Sessions Tab */}
          {activeTab === 'sessions' && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h1 className={styles.pageTitle}>
                  EEG Sessions for {selectedPatient?.full_name}
                </h1>
                <button
                  className={styles.primaryBtn}
                  onClick={handleUploadEEG}
                >
                  <Icons.Upload />
                  Upload New EEG
                </button>
              </div>

              {sessions.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.FileText />
                  <p>No EEG sessions for this patient</p>
                  <button className={styles.primaryBtn} onClick={handleUploadEEG}>
                    Upload First EEG
                  </button>
                </div>
              ) : (
                <div className={styles.sessionsList}>
                  {sessions.map(session => (
                    <div key={session.id} className={styles.sessionCard}>
                      <div className={styles.sessionHeader}>
                        <div>
                          <h3>Prediction ID: {session.id.substring(0, 8)}...</h3>
                          <p className={styles.filename}>{session.filename}</p>
                        </div>
                        <span className={`${styles.statusBadge} ${styles[session.status?.toLowerCase()]}`}>
                          {session.status}
                        </span>
                      </div>

                      <div className={styles.sessionDetails}>
                        <div className={styles.detailRow}>
                          <span>Date:</span>
                          <span>{new Date(session.created_at).toLocaleString()}</span>
                        </div>
                        {session.prediction && (
                          <div className={styles.detailRow}>
                            <span>Prediction:</span>
                            <strong style={{color: session.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981'}}>
                              {session.prediction}
                            </strong>
                          </div>
                        )}
                        {session.probabilities && session.probabilities.length > 0 && (
                          <div className={styles.detailRow}>
                            <span>Confidence:</span>
                            <strong>{(Math.max(...session.probabilities) * 100).toFixed(1)}%</strong>
                          </div>
                        )}
                      </div>

                      {session.status === 'Completed' && (
                        <div className={styles.sessionActions}>
                          {session.patient_pdf_url && (
                            <a href={session.patient_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.viewBtn}>
                              Patient Report
                            </a>
                          )}
                          {session.clinician_pdf_url && (
                            <a href={session.clinician_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.viewBtn}>
                              Clinician Report
                            </a>
                          )}
                          {session.technical_pdf_url && (
                            <a href={session.technical_pdf_url} target="_blank" rel="noopener noreferrer" className={styles.reportBtn}>
                              Technical Report
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>

        {/* Upload EEG Modal */}
        {showUploadModal && (
          <div className={styles.modal}>
            <div className={styles.modalContent}>
              <div className={styles.modalHeader}>
                <h2>Upload EEG File</h2>
                <button
                  className={styles.closeBtn}
                  onClick={() => setShowUploadModal(false)}
                  disabled={isUploading}
                >
                  ×
                </button>
              </div>

              <div className={styles.modalBody}>
                <div className={styles.uploadSection}>
                  <label className={styles.fileUploadLabel}>
                    <input
                      type="file"
                      accept=".npy"
                      onChange={handleFileSelect}
                      disabled={isUploading}
                      style={{ display: 'none' }}
                    />
                    <div className={styles.fileUploadBox}>
                      <Icons.Upload />
                      <p>{uploadData.file ? uploadData.file.name : 'Click to select .npy file'}</p>
                    </div>
                  </label>
                </div>

                <div className={styles.formGroup}>
                  <label>Select Channel (0-18)</label>
                  <select
                    value={uploadData.channelIndex}
                    onChange={(e) => setUploadData({ ...uploadData, channelIndex: parseInt(e.target.value) })}
                    disabled={isUploading}
                    className={styles.channelSelect}
                  >
                    {electrodeOptions.map((electrode, index) => (
                      <option key={electrode} value={index}>
                        Channel {index} - {electrode}
                      </option>
                    ))}
                  </select>
                  <small className={styles.helpText}>Select the EEG channel to use for similarity analysis visualization</small>
                </div>

                {isUploading && (
                  <div className={styles.progressBar}>
                    <div className={styles.progressFill} style={{ width: `${uploadProgress}%` }}></div>
                    <span>{uploadProgress}%</span>
                  </div>
                )}
              </div>

              <div className={styles.modalActions}>
                <button
                  className={styles.cancelBtn}
                  onClick={() => setShowUploadModal(false)}
                  disabled={isUploading}
                >
                  Cancel
                </button>
                <button
                  className={styles.submitBtn}
                  onClick={handleSubmitUpload}
                  disabled={isUploading || !uploadData.file}
                >
                  {isUploading ? 'Uploading...' : 'Upload & Analyze'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default withAuth(RadiologistDashboard, ['radiologist']);
