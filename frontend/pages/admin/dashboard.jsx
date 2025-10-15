import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import UnifiedSidebar from '../../components/UnifiedSidebar';
import LoadingSpinner from '../../components/LoadingSpinner';
import AddUserInterface from '../../components/admin/AddUserInterface';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/AdminDashboard.module.css';

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
  Users: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  ),
  UserPlus: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
      <circle cx="8.5" cy="7" r="4"/>
      <line x1="20" y1="8" x2="20" y2="14"/>
      <line x1="23" y1="11" x2="17" y2="11"/>
    </svg>
  ),
  CheckCircle: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>
  ),
  Heart: () => (
    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
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
  Menu: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="3" y1="12" x2="21" y2="12"/>
      <line x1="3" y1="6" x2="21" y2="6"/>
      <line x1="3" y1="18" x2="21" y2="18"/>
    </svg>
  ),
  ChevronLeft: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  ),
  AlertCircle: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  ),
};

function AdminDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const [dashboardStats, setDashboardStats] = useState({
    totalUsers: 0,
    pendingApprovals: 0,
    activePatients: 0,
    activeDoctors: 0,
    activeRadiologists: 0,
    unassignedPatients: 0,
  });

  const [pendingUsers, setPendingUsers] = useState([]);
  const [allPatients, setAllPatients] = useState([]);
  const [allDoctors, setAllDoctors] = useState([]);
  const [allRadiologists, setAllRadiologists] = useState([]);

  // Assignment modal states
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  // Search and filter states for each tab
  const [patientSearchTerm, setPatientSearchTerm] = useState('');
  const [doctorSearchTerm, setDoctorSearchTerm] = useState('');
  const [radiologistSearchTerm, setRadiologistSearchTerm] = useState('');
  const [patientFilter, setPatientFilter] = useState('all'); // all, assigned, unassigned
  const [doctorFilter, setDoctorFilter] = useState('all'); // all, with-patients, no-patients

  // Patient detail modal
  const [selectedPatientDetail, setSelectedPatientDetail] = useState(null);
  const [showPatientDetailModal, setShowPatientDetailModal] = useState(false);
  const [patientReports, setPatientReports] = useState([]);
  const [loadingReports, setLoadingReports] = useState(false);

  // Doctor detail modal
  const [selectedDoctorDetail, setSelectedDoctorDetail] = useState(null);
  const [showDoctorDetailModal, setShowDoctorDetailModal] = useState(false);
  const [doctorPatients, setDoctorPatients] = useState([]);
  const [loadingDoctorPatients, setLoadingDoctorPatients] = useState(false);

  // Radiologist detail modal
  const [selectedRadiologistDetail, setSelectedRadiologistDetail] = useState(null);
  const [showRadiologistDetailModal, setShowRadiologistDetailModal] = useState(false);
  const [radiologistActivities, setRadiologistActivities] = useState([]);
  const [loadingActivities, setLoadingActivities] = useState(false);

  // Approval modal
  const [selectedUser, setSelectedUser] = useState(null);
  const [showApprovalModal, setShowApprovalModal] = useState(false);

  const fetchDashboardData = useCallback(async () => {
    // Get hospital_id with fallback
    const hospitalId = userProfile?.hospital_id || hospitalData?.id;

    console.log('Admin Dashboard - Starting data fetch', {
      userProfile,
      hospitalData,
      hospitalId
    });

    try {
      setIsLoading(true);
      const { data: { session } } = await supabase.auth.getSession();

      console.log('Session status:', !!session?.access_token);

      if (!session?.access_token) {
        setError('Authentication required');
        setIsLoading(false);
        return;
      }

      console.log('Fetching from API...');

      const response = await fetch('/api/admin/users-simple', {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        }
      });

      console.log('API Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('API Result:', result);

      if (result.success && result.data) {
        const { pendingUsers, patients, doctors, radiologists, stats } = result.data;

        console.log('Dashboard data received:', {
          patientsCount: patients?.length,
          doctorsCount: doctors?.length,
          sampleDoctor: doctors?.[0],
          samplePatient: patients?.[0]
        });

        setPendingUsers(pendingUsers || []);
        setAllPatients(patients || []);
        setAllDoctors(doctors || []);
        setAllRadiologists(radiologists || []);

        setDashboardStats({
          totalUsers: stats?.totalUsers || 0,
          pendingApprovals: (stats?.pendingPatients || 0) + (stats?.pendingDoctors || 0) + (stats?.pendingRadiologists || 0),
          activePatients: stats?.activePatients || 0,
          activeDoctors: stats?.activeDoctors || 0,
          activeRadiologists: stats?.activeRadiologists || 0,
          unassignedPatients: (patients || []).filter(p => !p.patient_profiles?.[0]?.assigned_doctor_id).length,
        });
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setError('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  }, [userProfile]);

  useEffect(() => {
    if (user && userProfile && userProfile.role === 'admin') {
      fetchDashboardData();
    }
  }, [user, userProfile, fetchDashboardData]);

  const handleApproveUser = async (userId, role) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();

      const response = await fetch('/api/admin/approve', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId, role, action: 'approve' })
      });

      if (!response.ok) throw new Error('Approval failed');

      setShowApprovalModal(false);
      setSelectedUser(null);
      fetchDashboardData();
    } catch (error) {
      console.error('Approval error:', error);
      alert('Failed to approve user');
    }
  };

  const handleRejectUser = async (userId) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();

      const response = await fetch('/api/admin/approve', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId, action: 'reject' })
      });

      if (!response.ok) throw new Error('Rejection failed');

      setShowApprovalModal(false);
      setSelectedUser(null);
      fetchDashboardData();
    } catch (error) {
      console.error('Rejection error:', error);
      alert('Failed to reject user');
    }
  };

  const handleAssignDoctor = async () => {
    if (!selectedPatient || !selectedDoctor) return;

    try {
      const { data: { session } } = await supabase.auth.getSession();

      const response = await fetch('/api/admin/assign-doctor', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          patient_id: selectedPatient.id,
          doctor_id: selectedDoctor.id
        })
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to assign doctor');
      }

      alert(result.message || 'Doctor assigned successfully!');
      setShowAssignModal(false);
      setSelectedPatient(null);
      setSelectedDoctor(null);
      fetchDashboardData();
    } catch (error) {
      console.error('Assignment error:', error);
      alert(error.message || 'Failed to assign doctor');
    }
  };

  const fetchPatientReports = async (patientId) => {
    try {
      setLoadingReports(true);
      const { data, error } = await supabase
        .from('predictions')
        .select('*')
        .or(`patient_id.eq.${patientId},user_id.eq.${patientId}`)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setPatientReports(data || []);
    } catch (error) {
      console.error('Error fetching patient reports:', error);
      setPatientReports([]);
    } finally {
      setLoadingReports(false);
    }
  };

  const fetchDoctorPatients = async (doctorId) => {
    try {
      setLoadingDoctorPatients(true);
      const assignedPatients = allPatients.filter(patient => {
        const profile = Array.isArray(patient.patient_profiles)
          ? patient.patient_profiles[0]
          : patient.patient_profiles;
        return profile?.assigned_doctor_id === doctorId;
      });
      setDoctorPatients(assignedPatients);
    } catch (error) {
      console.error('Error fetching doctor patients:', error);
      setDoctorPatients([]);
    } finally {
      setLoadingDoctorPatients(false);
    }
  };

  const fetchRadiologistActivities = async (radiologistId) => {
    try {
      setLoadingActivities(true);
      const { data, error } = await supabase
        .from('predictions')
        .select('*')
        .eq('radiologist_id', radiologistId)
        .order('created_at', { ascending: false })
        .limit(20);

      if (error) throw error;
      setRadiologistActivities(data || []);
    } catch (error) {
      console.error('Error fetching radiologist activities:', error);
      setRadiologistActivities([]);
    } finally {
      setLoadingActivities(false);
    }
  };

  const filteredDoctors = allDoctors.filter(doc =>
    doc.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.doctor_profiles?.[0]?.specialization?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Filter patients based on search and filter
  const filteredPatients = allPatients.filter(patient => {
    const profile = Array.isArray(patient.patient_profiles) ? patient.patient_profiles[0] : patient.patient_profiles;
    const matchesSearch = patient.full_name?.toLowerCase().includes(patientSearchTerm.toLowerCase()) ||
      patient.unique_identifier?.toLowerCase().includes(patientSearchTerm.toLowerCase()) ||
      patient.phone?.includes(patientSearchTerm);

    if (patientFilter === 'assigned') {
      return matchesSearch && profile?.assigned_doctor_id;
    } else if (patientFilter === 'unassigned') {
      return matchesSearch && !profile?.assigned_doctor_id;
    }
    return matchesSearch;
  });

  // Filter doctors based on search and filter
  const filteredDoctorsList = allDoctors.filter(doctor => {
    const profile = Array.isArray(doctor.doctor_profiles) ? doctor.doctor_profiles[0] : doctor.doctor_profiles;
    const matchesSearch = doctor.full_name?.toLowerCase().includes(doctorSearchTerm.toLowerCase()) ||
      doctor.unique_identifier?.toLowerCase().includes(doctorSearchTerm.toLowerCase()) ||
      profile?.specialization?.toLowerCase().includes(doctorSearchTerm.toLowerCase());

    if (doctorFilter === 'with-patients') {
      const hasPatients = allPatients.some(p => {
        const pProfile = Array.isArray(p.patient_profiles) ? p.patient_profiles[0] : p.patient_profiles;
        return pProfile?.assigned_doctor_id === doctor.id;
      });
      return matchesSearch && hasPatients;
    } else if (doctorFilter === 'no-patients') {
      const hasPatients = allPatients.some(p => {
        const pProfile = Array.isArray(p.patient_profiles) ? p.patient_profiles[0] : p.patient_profiles;
        return pProfile?.assigned_doctor_id === doctor.id;
      });
      return matchesSearch && !hasPatients;
    }
    return matchesSearch;
  });

  // Filter radiologists based on search
  const filteredRadiologists = allRadiologists.filter(radiologist => {
    return radiologist.full_name?.toLowerCase().includes(radiologistSearchTerm.toLowerCase()) ||
      radiologist.unique_identifier?.toLowerCase().includes(radiologistSearchTerm.toLowerCase()) ||
      radiologist.email?.toLowerCase().includes(radiologistSearchTerm.toLowerCase());
  });

  const navigationItems = [
    { id: 'overview', label: 'Dashboard', icon: 'Dashboard' },
    { id: 'add-user', label: 'Add User', icon: 'UserPlus' },
    { id: 'patients', label: 'Patients', icon: 'Heart', badgeKey: 'activePatients' },
    { id: 'doctors', label: 'Doctors', icon: 'Stethoscope', badgeKey: 'activeDoctors' },
    { id: 'radiologists', label: 'Radiologists', icon: 'Activity', badgeKey: 'activeRadiologists' },
  ];

  if (isLoading && !userProfile) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading dashboard...</p>
      </div>
    );
  }

  if (error && !isLoading) {
    return (
      <>
        <Navbar />
        <div className={styles.dashboardContainer}>
          <div className={styles.errorState}>
            <h2>Error Loading Dashboard</h2>
            <p>{error}</p>
            <button onClick={() => {
              setError('');
              fetchDashboardData();
            }} className={styles.primaryButton}>
              Retry
            </button>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />

      <div className={styles.dashboardContainer}>
        <UnifiedSidebar
          user={user}
          userProfile={userProfile}
          hospitalData={hospitalData}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          navigationItems={navigationItems}
          stats={dashboardStats}
        />

        {/* Main Content */}
        <main className={styles.mainContent}>
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className={styles.overviewSection}>
              <h1 className={styles.pageTitle}>Dashboard Overview</h1>

              <div className={styles.statsGrid}>
                <div className={styles.statCard} onClick={() => setActiveTab('patients')}>
                  <Icons.Users />
                  <div>
                    <h3>Total Users</h3>
                    <p className={styles.statNumber}>{dashboardStats.totalUsers}</p>
                  </div>
                </div>

                <div className={styles.statCard} onClick={() => setActiveTab('patients')}>
                  <Icons.Heart />
                  <div>
                    <h3>Active Patients</h3>
                    <p className={styles.statNumber}>{dashboardStats.activePatients}</p>
                  </div>
                </div>

                <div className={styles.statCard} onClick={() => setActiveTab('doctors')}>
                  <Icons.Stethoscope />
                  <div>
                    <h3>Active Doctors</h3>
                    <p className={styles.statNumber}>{dashboardStats.activeDoctors}</p>
                  </div>
                </div>

                <div className={styles.statCard} onClick={() => setActiveTab('radiologists')}>
                  <Icons.Activity />
                  <div>
                    <h3>Radiologists</h3>
                    <p className={styles.statNumber}>{dashboardStats.activeRadiologists}</p>
                  </div>
                </div>

                <div className={styles.statCard} onClick={() => setActiveTab('patients')}>
                  <Icons.AlertCircle />
                  <div>
                    <h3>Unassigned Patients</h3>
                    <p className={styles.statNumber}>{dashboardStats.unassignedPatients}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Add User Tab */}
          {activeTab === 'add-user' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Add New User</h1>
              <AddUserInterface onUserCreated={fetchDashboardData} />
            </div>
          )}

          {/* Patients Tab - Redesigned with Search and Filters */}
          {activeTab === 'patients' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Patient Management</h1>

              {/* Search and Filter Bar */}
              <div className={styles.filterBar}>
                <div className={styles.searchBox}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.35-4.35"/>
                  </svg>
                  <input
                    type="text"
                    placeholder="Search by name, ID, or phone..."
                    value={patientSearchTerm}
                    onChange={(e) => setPatientSearchTerm(e.target.value)}
                    className={styles.searchInputBar}
                  />
                </div>
                <div className={styles.filterButtons}>
                  <button
                    className={`${styles.filterBtn} ${patientFilter === 'all' ? styles.filterBtnActive : ''}`}
                    onClick={() => setPatientFilter('all')}
                  >
                    All Patients ({allPatients.length})
                  </button>
                  <button
                    className={`${styles.filterBtn} ${patientFilter === 'unassigned' ? styles.filterBtnActive : ''}`}
                    onClick={() => setPatientFilter('unassigned')}
                  >
                    Unassigned ({allPatients.filter(p => {
                      const profile = Array.isArray(p.patient_profiles) ? p.patient_profiles[0] : p.patient_profiles;
                      return !profile?.assigned_doctor_id;
                    }).length})
                  </button>
                  <button
                    className={`${styles.filterBtn} ${patientFilter === 'assigned' ? styles.filterBtnActive : ''}`}
                    onClick={() => setPatientFilter('assigned')}
                  >
                    Assigned ({allPatients.filter(p => {
                      const profile = Array.isArray(p.patient_profiles) ? p.patient_profiles[0] : p.patient_profiles;
                      return profile?.assigned_doctor_id;
                    }).length})
                  </button>
                </div>
              </div>

              {!allPatients || allPatients.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.Heart />
                  <p>No patients found</p>
                  <button onClick={() => setActiveTab('add-user')} className={styles.primaryBtn} style={{width: 'auto', marginTop: '1rem'}}>
                    Add Patient
                  </button>
                </div>
              ) : filteredPatients.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.Heart />
                  <p>No patients match your search</p>
                </div>
              ) : (
                <>
                  <div className={styles.cardGrid}>
                      {filteredPatients
                        .map(patient => {
                          const profile = Array.isArray(patient.patient_profiles)
                            ? patient.patient_profiles[0]
                            : patient.patient_profiles;

                          return (
                            <div key={patient.id} className={styles.patientCard}>
                              <div
                                className={styles.cardHeader}
                                onClick={() => {
                                  setSelectedPatientDetail({ ...patient, profile });
                                  setShowPatientDetailModal(true);
                                  fetchPatientReports(patient.id);
                                }}
                                style={{ cursor: 'pointer' }}
                              >
                                <h3>{patient.full_name}</h3>
                                <span className={styles.patientId}>{patient.unique_identifier}</span>
                              </div>

                              <div className={styles.cardBody}>
                                <p><strong>Phone:</strong> {patient.phone}</p>
                                <p><strong>Blood Group:</strong> {profile?.blood_groups?.blood_type || 'N/A'}</p>
                                <p><strong>Age:</strong> {patient.date_of_birth ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear() : 'N/A'} years</p>

                                {/* Check if patient has assigned doctor */}
                                {profile?.assigned_doctor_id ? (
                                  <div className={styles.assignedInfo}>
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                                      <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
                                      <circle cx="20" cy="10" r="2"/>
                                    </svg>
                                    <div>
                                      <strong>Assigned Doctor</strong>
                                      <p>{allDoctors.find(d => d.id === profile.assigned_doctor_id)?.full_name || 'Unknown'}</p>
                                    </div>
                                  </div>
                                ) : (
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setSelectedPatient(patient);
                                      setShowAssignModal(true);
                                    }}
                                    className={styles.assignBtn}
                                  >
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                      <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                                      <circle cx="8.5" cy="7" r="4"/>
                                      <line x1="20" y1="8" x2="20" y2="14"/>
                                      <line x1="23" y1="11" x2="17" y2="11"/>
                                    </svg>
                                    Assign Doctor
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                    </div>
                </>
              )}
            </div>
          )}

          {/* Doctors Tab */}
          {activeTab === 'doctors' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Doctor Management</h1>

              {/* Search and Filter Bar */}
              <div className={styles.filterBar}>
                <div className={styles.searchBox}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.35-4.35"/>
                  </svg>
                  <input
                    type="text"
                    placeholder="Search by name, ID, or specialization..."
                    value={doctorSearchTerm}
                    onChange={(e) => setDoctorSearchTerm(e.target.value)}
                    className={styles.searchInputBar}
                  />
                </div>
                <div className={styles.filterButtons}>
                  <button
                    className={`${styles.filterBtn} ${doctorFilter === 'all' ? styles.filterBtnActive : ''}`}
                    onClick={() => setDoctorFilter('all')}
                  >
                    All Doctors ({allDoctors.length})
                  </button>
                  <button
                    className={`${styles.filterBtn} ${doctorFilter === 'with-patients' ? styles.filterBtnActive : ''}`}
                    onClick={() => setDoctorFilter('with-patients')}
                  >
                    With Patients
                  </button>
                  <button
                    className={`${styles.filterBtn} ${doctorFilter === 'no-patients' ? styles.filterBtnActive : ''}`}
                    onClick={() => setDoctorFilter('no-patients')}
                  >
                    No Patients
                  </button>
                </div>
              </div>

              {!allDoctors || allDoctors.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.Stethoscope />
                  <p>No doctors found</p>
                  <button onClick={() => setActiveTab('add-user')} className={styles.primaryBtn} style={{width: 'auto', marginTop: '1rem'}}>
                    Add Doctor
                  </button>
                </div>
              ) : filteredDoctorsList.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.Stethoscope />
                  <p>No doctors match your search</p>
                </div>
              ) : (
                <div className={styles.cardGrid}>
                  {filteredDoctorsList.map(doctor => {
                    // Handle both array and single object for doctor_profiles
                    const profile = Array.isArray(doctor.doctor_profiles)
                      ? doctor.doctor_profiles[0]
                      : doctor.doctor_profiles;

                    return (
                      <div key={doctor.id} className={styles.doctorCard}>
                        <div
                          className={styles.cardHeader}
                          onClick={() => {
                            setSelectedDoctorDetail({ ...doctor, profile });
                            setShowDoctorDetailModal(true);
                            fetchDoctorPatients(doctor.id);
                          }}
                          style={{ cursor: 'pointer' }}
                        >
                          <h3>{doctor.full_name || 'Unknown'}</h3>
                          <span className={styles.doctorId}>{doctor.unique_identifier || 'N/A'}</span>
                        </div>

                        <div className={styles.cardBody}>
                          <p><strong>Email:</strong> {doctor.email || 'N/A'}</p>
                          <p><strong>Phone:</strong> {doctor.phone || 'N/A'}</p>
                          <p><strong>License:</strong> {profile?.medical_license || 'N/A'}</p>
                          <p><strong>Specialization:</strong> {profile?.specialization || 'General'}</p>
                          <p><strong>Experience:</strong> {profile?.experience_years || 0} years</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Radiologists Tab */}
          {activeTab === 'radiologists' && (
            <div className={styles.section}>
              <h1 className={styles.pageTitle}>Radiologist Management</h1>

              {allRadiologists.length === 0 ? (
                <div className={styles.emptyState}>
                  <Icons.Activity />
                  <p>No radiologists found</p>
                </div>
              ) : (
                <div className={styles.cardGrid}>
                  {allRadiologists.map(radiologist => (
                    <div key={radiologist.id} className={styles.radiologistCard}>
                      <div
                        className={styles.cardHeader}
                        onClick={() => {
                          setSelectedRadiologistDetail(radiologist);
                          setShowRadiologistDetailModal(true);
                          fetchRadiologistActivities(radiologist.id);
                        }}
                        style={{ cursor: 'pointer' }}
                      >
                        <h3>{radiologist.full_name}</h3>
                        <span className={styles.radiologistId}>{radiologist.unique_identifier}</span>
                      </div>

                      <div className={styles.cardBody}>
                        <p><strong>Email:</strong> {radiologist.email}</p>
                        <p><strong>Phone:</strong> {radiologist.phone}</p>
                        <p><strong>Status:</strong> {radiologist.account_status}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Assignment Modal */}
      {showAssignModal && selectedPatient && (
        <div className={styles.modal} onClick={() => setShowAssignModal(false)}>
          <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Assign Doctor to {selectedPatient.full_name}</h2>
              <button onClick={() => setShowAssignModal(false)} className={styles.closeBtn}>×</button>
            </div>

            <input
              type="text"
              placeholder="Search doctors..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className={styles.searchInput}
            />

            <div className={styles.doctorList}>
              {filteredDoctors.map(doctor => (
                <div
                  key={doctor.id}
                  onClick={() => setSelectedDoctor(doctor)}
                  className={`${styles.doctorOption} ${selectedDoctor?.id === doctor.id ? styles.selected : ''}`}
                >
                  <h4>{doctor.full_name}</h4>
                  <p>{doctor.doctor_profiles?.[0]?.specialization || 'General Medicine'}</p>
                  <span>{doctor.doctor_profiles?.[0]?.experience_years || 0} years exp</span>
                </div>
              ))}
            </div>

            {selectedDoctor && (
              <button onClick={handleAssignDoctor} className={styles.primaryBtn}>
                Assign {selectedDoctor.full_name}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Patient Detail Modal with Reports */}
      {showPatientDetailModal && selectedPatientDetail && (
        <div className={styles.modal} onClick={() => {
          setShowPatientDetailModal(false);
          setPatientReports([]);
        }}>
          <div className={styles.modalContentLarge} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Patient Details</h2>
              <button onClick={() => {
                setShowPatientDetailModal(false);
                setPatientReports([]);
              }} className={styles.closeBtn}>×</button>
            </div>

            <div className={styles.modalBody}>
              <div className={styles.userDetails}>
                <div className={styles.avatarLarge}>
                  {selectedPatientDetail.full_name?.[0]?.toUpperCase()}
                </div>
                <h3>{selectedPatientDetail.full_name}</h3>
                <span className={styles.roleTag}>Patient</span>
              </div>

              <div className={styles.detailsGrid}>
                <div><strong>Patient ID:</strong> {selectedPatientDetail.unique_identifier}</div>
                <div><strong>Email:</strong> {selectedPatientDetail.email || 'N/A'}</div>
                <div><strong>Phone:</strong> {selectedPatientDetail.phone}</div>
                <div><strong>Age:</strong> {selectedPatientDetail.date_of_birth ? new Date().getFullYear() - new Date(selectedPatientDetail.date_of_birth).getFullYear() : 'N/A'} years</div>
                <div><strong>Blood Group:</strong> {selectedPatientDetail.profile?.blood_groups?.blood_type || 'N/A'}</div>
                <div><strong>Account Status:</strong> {selectedPatientDetail.account_status}</div>
              </div>

              {selectedPatientDetail.assignedDoctor && (
                <div className={styles.assignedDoctorSection}>
                  <h4>Assigned Doctor</h4>
                  <div className={styles.doctorInfo}>
                    <p><strong>Name:</strong> {selectedPatientDetail.assignedDoctor.full_name}</p>
                    <p><strong>Email:</strong> {selectedPatientDetail.assignedDoctor.email}</p>
                    <p><strong>Phone:</strong> {selectedPatientDetail.assignedDoctor.phone || 'N/A'}</p>
                  </div>
                </div>
              )}

              <div className={styles.reportsSection}>
                <h4>Patient Reports ({patientReports.length})</h4>
                {loadingReports ? (
                  <p>Loading reports...</p>
                ) : patientReports.length > 0 ? (
                  <div className={styles.reportsListModal}>
                    {patientReports.map(report => (
                      <div key={report.id} className={styles.reportItemModal}>
                        <div className={styles.reportHeaderModal}>
                          <h5>{report.session_code || `Session-${report.id.substring(0, 8)}`}</h5>
                          <span className={styles.statusBadge}>{report.status}</span>
                        </div>
                        <p><strong>Date:</strong> {new Date(report.created_at).toLocaleDateString()}</p>
                        <p><strong>Filename:</strong> {report.filename}</p>
                        {report.prediction && (
                          <p><strong>Result:</strong> <span style={{color: report.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981'}}>{report.prediction}</span></p>
                        )}
                        {report.patient_pdf_url && (
                          <button
                            onClick={() => window.open(report.patient_pdf_url, '_blank')}
                            className={styles.viewReportBtnSmall}
                          >
                            View Report
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{color: '#666'}}>No reports available</p>
                )}
              </div>
            </div>

            <div className={styles.modalActions}>
              <button
                onClick={() => {
                  setShowPatientDetailModal(false);
                  setPatientReports([]);
                }}
                className={styles.primaryBtn}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Doctor Detail Modal with Assigned Patients */}
      {showDoctorDetailModal && selectedDoctorDetail && (
        <div className={styles.modal} onClick={() => {
          setShowDoctorDetailModal(false);
          setDoctorPatients([]);
        }}>
          <div className={styles.modalContentLarge} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Doctor Details</h2>
              <button onClick={() => {
                setShowDoctorDetailModal(false);
                setDoctorPatients([]);
              }} className={styles.closeBtn}>×</button>
            </div>

            <div className={styles.modalBody}>
              <div className={styles.userDetails}>
                <div className={styles.avatarLarge}>
                  {selectedDoctorDetail.full_name?.[0]?.toUpperCase()}
                </div>
                <h3>{selectedDoctorDetail.full_name}</h3>
                <span className={styles.roleTag}>Doctor</span>
              </div>

              <div className={styles.detailsGrid}>
                <div><strong>Doctor ID:</strong> {selectedDoctorDetail.unique_identifier}</div>
                <div><strong>Email:</strong> {selectedDoctorDetail.email || 'N/A'}</div>
                <div><strong>Phone:</strong> {selectedDoctorDetail.phone || 'N/A'}</div>
                <div><strong>Medical License:</strong> {selectedDoctorDetail.profile?.medical_license || 'N/A'}</div>
                <div><strong>Specialization:</strong> {selectedDoctorDetail.profile?.specialization || 'General'}</div>
                <div><strong>Experience:</strong> {selectedDoctorDetail.profile?.experience_years || 0} years</div>
              </div>

              <div className={styles.reportsSection}>
                <h4>Assigned Patients ({doctorPatients.length})</h4>
                {loadingDoctorPatients ? (
                  <p>Loading patients...</p>
                ) : doctorPatients.length > 0 ? (
                  <div className={styles.reportsListModal}>
                    {doctorPatients.map(patient => {
                      const profile = Array.isArray(patient.patient_profiles)
                        ? patient.patient_profiles[0]
                        : patient.patient_profiles;

                      return (
                        <div key={patient.id} className={styles.reportItemModal}>
                          <div className={styles.reportHeaderModal}>
                            <h5>{patient.full_name}</h5>
                            <span className={styles.statusBadge}>{patient.account_status}</span>
                          </div>
                          <p><strong>Patient ID:</strong> {patient.unique_identifier}</p>
                          <p><strong>Phone:</strong> {patient.phone}</p>
                          <p><strong>Age:</strong> {patient.date_of_birth ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear() : 'N/A'} years</p>
                          <p><strong>Blood Group:</strong> {profile?.blood_groups?.blood_type || 'N/A'}</p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p style={{color: '#666'}}>No patients assigned yet</p>
                )}
              </div>
            </div>

            <div className={styles.modalActions}>
              <button
                onClick={() => {
                  setShowDoctorDetailModal(false);
                  setDoctorPatients([]);
                }}
                className={styles.primaryBtn}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Radiologist Detail Modal with Activities */}
      {showRadiologistDetailModal && selectedRadiologistDetail && (
        <div className={styles.modal} onClick={() => {
          setShowRadiologistDetailModal(false);
          setRadiologistActivities([]);
        }}>
          <div className={styles.modalContentLarge} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Radiologist Details</h2>
              <button onClick={() => {
                setShowRadiologistDetailModal(false);
                setRadiologistActivities([]);
              }} className={styles.closeBtn}>×</button>
            </div>

            <div className={styles.modalBody}>
              <div className={styles.userDetails}>
                <div className={styles.avatarLarge}>
                  {selectedRadiologistDetail.full_name?.[0]?.toUpperCase()}
                </div>
                <h3>{selectedRadiologistDetail.full_name}</h3>
                <span className={styles.roleTag}>Radiologist</span>
              </div>

              <div className={styles.detailsGrid}>
                <div><strong>Radiologist ID:</strong> {selectedRadiologistDetail.unique_identifier}</div>
                <div><strong>Email:</strong> {selectedRadiologistDetail.email}</div>
                <div><strong>Phone:</strong> {selectedRadiologistDetail.phone}</div>
                <div><strong>Account Status:</strong> {selectedRadiologistDetail.account_status}</div>
              </div>

              <div className={styles.reportsSection}>
                <h4>Recent Activities & Analysis ({radiologistActivities.length})</h4>
                {loadingActivities ? (
                  <p>Loading activities...</p>
                ) : radiologistActivities.length > 0 ? (
                  <div className={styles.reportsListModal}>
                    {radiologistActivities.map(activity => (
                      <div key={activity.id} className={styles.reportItemModal}>
                        <div className={styles.reportHeaderModal}>
                          <h5>{activity.session_code || `Session-${activity.id.substring(0, 8)}`}</h5>
                          <span className={styles.statusBadge}>{activity.status}</span>
                        </div>
                        <p><strong>Date:</strong> {new Date(activity.created_at).toLocaleDateString()} {new Date(activity.created_at).toLocaleTimeString()}</p>
                        <p><strong>File:</strong> {activity.filename}</p>
                        {activity.prediction && (
                          <p><strong>Analysis Result:</strong> <span style={{color: activity.prediction.toLowerCase().includes('alz') ? '#ef4444' : '#10b981'}}>{activity.prediction}</span></p>
                        )}
                        {activity.confidence && (
                          <p><strong>Confidence:</strong> {(activity.confidence * 100).toFixed(1)}%</p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{color: '#666'}}>No recent activities found</p>
                )}
              </div>
            </div>

            <div className={styles.modalActions}>
              <button
                onClick={() => {
                  setShowRadiologistDetailModal(false);
                  setRadiologistActivities([]);
                }}
                className={styles.primaryBtn}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Approval Modal */}
      {showApprovalModal && selectedUser && (
        <div className={styles.modal} onClick={() => setShowApprovalModal(false)}>
          <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Review Application</h2>
              <button onClick={() => setShowApprovalModal(false)} className={styles.closeBtn}>×</button>
            </div>

            <div className={styles.modalBody}>
              <div className={styles.userDetails}>
                <div className={styles.avatarLarge}>{selectedUser.full_name?.[0]?.toUpperCase()}</div>
                <h3>{selectedUser.full_name}</h3>
                <span className={`${styles.roleTag} ${styles[selectedUser.role]}`}>{selectedUser.role}</span>
              </div>

              <div className={styles.detailsGrid}>
                <div><strong>Email:</strong> {selectedUser.email}</div>
                <div><strong>Phone:</strong> {selectedUser.phone}</div>
                <div><strong>Address:</strong> {selectedUser.address}</div>
                <div><strong>ID:</strong> {selectedUser.unique_identifier}</div>
              </div>
            </div>

            <div className={styles.modalActions}>
              <button
                onClick={() => handleApproveUser(selectedUser.id, selectedUser.role)}
                className={styles.approveBtn}
              >
                Approve
              </button>
              <button
                onClick={() => handleRejectUser(selectedUser.id)}
                className={styles.rejectBtn}
              >
                Reject
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default withAuth(AdminDashboard, ['admin']);