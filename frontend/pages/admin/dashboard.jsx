import { useState, useEffect } from 'react';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/DashboardLayout.module.css';

function AdminDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardStats, setDashboardStats] = useState({
    totalUsers: 0,
    pendingPatients: 0,
    pendingDoctors: 0,
    activePatients: 0,
    activeDoctors: 0,
    unassignedPatients: 0
  });
  
  // User management state
  const [pendingUsers, setPendingUsers] = useState([]);
  const [allPatients, setAllPatients] = useState([]);
  const [allDoctors, setAllDoctors] = useState([]);
  const [assignmentMode, setAssignmentMode] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showUserModal, setShowUserModal] = useState(false);
  const [selectedUserForApproval, setSelectedUserForApproval] = useState(null);

  useEffect(() => {
    if (userProfile && hospitalData) {
      fetchDashboardData();
    } else {
      if (userProfile && !hospitalData && userProfile.hospital_id) {
        fetchHospitalData();
      } else if (userProfile && userProfile.role === 'admin') {
        setIsLoading(false);
      }
    }
  }, [userProfile, hospitalData]);

  const fetchHospitalData = async () => {
    try {
      const { data, error } = await supabase
        .from('hospitals')
        .select('*')
        .eq('id', userProfile.hospital_id)
        .single();

      if (error) {
        setIsLoading(false);
        return;
      }

      if (data) {
        fetchDashboardData();
      }
    } catch (error) {
      setIsLoading(false);
    }
  };

  const fetchDashboardData = async (forceRefresh = false) => {
    try {
      setIsLoading(true);

      // ENTERPRISE FIX: Add caching to improve performance
      const cacheKey = `admin_dashboard_${userProfile?.hospital_id}`;
      const cachedData = sessionStorage.getItem(cacheKey);
      const cacheTimestamp = sessionStorage.getItem(`${cacheKey}_timestamp`);
      
      // Use cache if data is less than 30 seconds old and not forcing refresh
      if (!forceRefresh && cachedData && cacheTimestamp) {
        const age = Date.now() - parseInt(cacheTimestamp);
        if (age < 30000) {
          const parsed = JSON.parse(cachedData);
          setPendingUsers(parsed.pendingUsers || []);
          setAllPatients(parsed.patients || []);
          setAllDoctors(parsed.doctors || []);
          setDashboardStats(parsed.stats || {});
          setIsLoading(false);
          return;
        }
      }

      // Get the current session token for API authentication
      const { data: { session } } = await supabase.auth.getSession();
      if (!session?.access_token) {
        setIsLoading(false);
        return;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch('/api/admin/users', {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}: Failed to fetch admin data`);
      }

      const result = await response.json();

      if (result.success && result.data) {
        const { pendingUsers, patients, doctors, stats } = result.data;
        
        setPendingUsers(pendingUsers || []);
        setAllPatients(patients || []);
        setAllDoctors(doctors || []);
        setDashboardStats(stats || {});
        
        sessionStorage.setItem(cacheKey, JSON.stringify(result.data));
        sessionStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
        
      } else {
        throw new Error('Invalid API response format');
      }

    } catch (error) {
      if (error.name === 'AbortError') {
        alert('Request timed out. Please check your network connection and try again.');
      } else {
        alert('Failed to load dashboard data. Please refresh the page.');
      }
    } finally {
      setIsLoading(false);
    }
  };


  const handleViewUserDetails = (user) => {
    setSelectedUserForApproval(user);
    setShowUserModal(true);
  };

  const handleApproveUser = async (userId, role) => {
    try {
      setIsLoading(true);
      
      const userBefore = pendingUsers.find(u => u.id === userId);
      
      const { data: { session } } = await supabase.auth.getSession();
      
      const response = await fetch('/api/admin/approve', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId, role, action: 'approve' })
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to approve user');
      }

      await new Promise(resolve => setTimeout(resolve, 1000));

      setPendingUsers(prev => prev.filter(u => u.id !== userId));
      
      setDashboardStats(prev => ({
        ...prev,
        pendingPatients: role === 'patient' ? prev.pendingPatients - 1 : prev.pendingPatients,
        pendingDoctors: role === 'doctor' ? prev.pendingDoctors - 1 : prev.pendingDoctors,
        activePatients: role === 'patient' ? prev.activePatients + 1 : prev.activePatients,
        activeDoctors: role === 'doctor' ? prev.activeDoctors + 1 : prev.activeDoctors,
      }));

      setShowUserModal(false);
      setSelectedUserForApproval(null);
      
      const cacheKey = `admin_dashboard_${userProfile?.hospital_id}`;
      sessionStorage.removeItem(cacheKey);
      sessionStorage.removeItem(`${cacheKey}_timestamp`);
      
      await fetchDashboardData(true);
      
      alert(`User ${userBefore?.full_name} approved successfully!`);
      
    } catch (error) {
      alert(`Error approving user: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRejectUser = async (userId) => {
    try {
      const user = pendingUsers.find(u => u.id === userId);
      
      if (!confirm(`Are you sure you want to reject ${user?.full_name || 'this user'}?`)) {
        return;
      }

      setIsLoading(true);
      
      const { data: { session } } = await supabase.auth.getSession();
      
      const response = await fetch('/api/admin/approve', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId, role: user.role, action: 'reject' })
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to reject user');
      }

      setPendingUsers(prev => prev.filter(u => u.id !== userId));
      
      setDashboardStats(prev => ({
        ...prev,
        pendingPatients: user.role === 'patient' ? prev.pendingPatients - 1 : prev.pendingPatients,
        pendingDoctors: user.role === 'doctor' ? prev.pendingDoctors - 1 : prev.pendingDoctors,
      }));

      setShowUserModal(false);
      setSelectedUserForApproval(null);
      
      const cacheKey = `admin_dashboard_${userProfile?.hospital_id}`;
      sessionStorage.removeItem(cacheKey);
      sessionStorage.removeItem(`${cacheKey}_timestamp`);
      
    } catch (error) {
      alert('Error rejecting user. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAssignDoctor = async (patientId, doctorId) => {
    try {
      const { error } = await supabase
        .from('patient_profiles')
        .update({ 
          assigned_doctor_id: doctorId,
          updated_at: new Date().toISOString()
        })
        .eq('user_id', patientId);

      if (error) throw error;
      
      fetchDashboardData();
      setAssignmentMode(null);
      setSelectedPatient(null);
      alert('Doctor assigned successfully!');
    } catch (error) {
      alert('Error assigning doctor. Please try again.');
    }
  };


  const renderOverview = () => (
    <div className={styles.overviewGrid}>
      <div 
        className={styles.statCard}
        onClick={() => {
          setActiveTab('patients');
        }}
      >
        <div className={styles.statIcon}>👥</div>
        <div className={styles.statContent}>
          <h3>Total Users</h3>
          <div className={styles.statNumber}>{dashboardStats.totalUsers}</div>
          <p>In your hospital • Click to view all</p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => {
          setActiveTab('approvals');
        }}
      >
        <div className={styles.statIcon}>⏳</div>
        <div className={styles.statContent}>
          <h3>Pending Approvals</h3>
          <div className={styles.statNumber}>{dashboardStats.pendingPatients + dashboardStats.pendingDoctors}</div>
          <p>{dashboardStats.pendingPatients} patients, {dashboardStats.pendingDoctors} doctors • Click to review</p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => {
          setActiveTab('patients');
        }}
      >
        <div className={styles.statIcon}>🏥</div>
        <div className={styles.statContent}>
          <h3>Active Users</h3>
          <div className={styles.statNumber}>{dashboardStats.activePatients + dashboardStats.activeDoctors}</div>
          <p>{dashboardStats.activePatients} patients, {dashboardStats.activeDoctors} doctors • Click to manage</p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => {
          setActiveTab('patients');
        }}
      >
        <div className={styles.statIcon}>👨‍⚕️</div>
        <div className={styles.statContent}>
          <h3>Unassigned Patients</h3>
          <div className={styles.statNumber}>{dashboardStats.unassignedPatients}</div>
          <p>Need doctor assignment • Click to assign</p>
        </div>
      </div>
    </div>
  );

  const renderPendingApprovals = () => (
    <div className={styles.approvalSection}>
      <div className={styles.sectionHeader}>
        <h2>🔍 User Approvals</h2>
        <div className={styles.statsChips}>
          <span className={styles.chip}>
            {pendingUsers.length} Pending
          </span>
          <button 
            onClick={() => fetchDashboardData(true)}
            className={styles.refreshButton}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>
      
      {pendingUsers.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            {isLoading ? '🔄' : '✨'}
          </div>
          <h3>{isLoading ? 'Loading...' : 'All caught up!'}</h3>
          <p>
            {isLoading ? 'Fetching pending approvals...' : 'No users waiting for approval'}
          </p>
          {!isLoading && (
            <div className={styles.emptyStateActions}>
              <button 
                onClick={() => fetchDashboardData(true)}
                className={styles.refreshDataButton}
                disabled={isLoading}
              >
                {isLoading ? 'Loading...' : 'Refresh Data'}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className={styles.approvalList}>
          {pendingUsers.map(user => (
            <div key={user.id} className={styles.approvalCard}>
              <div className={styles.userInfo}>
                <div className={styles.avatar}>
                  {user.full_name?.charAt(0)?.toUpperCase()}
                </div>
                <div className={styles.userDetails}>
                  <div className={styles.nameRow}>
                    <h3>{user.full_name}</h3>
                    <span className={`${styles.roleChip} ${styles[user.role]}`}>
                      {user.role}
                    </span>
                  </div>
                  <div className={styles.contactRow}>
                    <span className={styles.email}>{user.email}</span>
                    <span className={styles.phone}>{user.phone}</span>
                  </div>
                  <div className={styles.idRow}>
                    <span className={styles.userId}>{user.unique_identifier}</span>
                    <span className={styles.joinDate}>
                      {new Date(user.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className={styles.actionButtons}>
                <button 
                  onClick={() => handleViewUserDetails(user)}
                  className={styles.viewButton}
                >
                  Review Application
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderPatientManagement = () => (
    <div className={styles.patientManagement}>
      <div className={styles.managementHeader}>
        <h2>Patient Management</h2>
        <div className={styles.patientStats}>
          <span>Total: {allPatients.length}</span>
          <span>Unassigned: {dashboardStats.unassignedPatients}</span>
        </div>
      </div>
      
      {assignmentMode && (
        <div className={styles.assignmentModal}>
          <div className={styles.modalContent}>
            <h3>Assign Doctor to {selectedPatient?.full_name}</h3>
            <div className={styles.doctorList}>
              {allDoctors.map(doctor => (
                <div key={doctor.id} className={styles.doctorOption} onClick={() => handleAssignDoctor(selectedPatient.id, doctor.id)}>
                  <div className={styles.doctorInfo}>
                    <h4>{doctor.full_name}</h4>
                    <p>{doctor.doctor_profiles?.[0]?.specialization}</p>
                    <p>{doctor.doctor_profiles?.[0]?.experience_years} years experience</p>
                    <p>Current patients: {doctor.doctor_profiles?.[0]?.patient_count || 0}</p>
                  </div>
                </div>
              ))}
            </div>
            <button onClick={() => { setAssignmentMode(false); setSelectedPatient(null); }} className={styles.cancelBtn}>Cancel</button>
          </div>
        </div>
      )}
      
      <div className={styles.patientGrid}>
        {allPatients.map(patient => (
          <div key={patient.id} className={styles.patientCard}>
            <div className={styles.patientHeader}>
              <h3>{patient.full_name}</h3>
              <span className={styles.patientId}>{patient.unique_identifier}</span>
            </div>
            
            <div className={styles.patientDetails}>
              <p><strong>Phone:</strong> {patient.phone}</p>
              <p><strong>Blood Group:</strong> {patient.patient_profiles?.[0]?.blood_groups?.blood_type || 'N/A'}</p>
              <p><strong>Emergency Contact:</strong> {patient.patient_profiles?.[0]?.emergency_contact_name || 'N/A'}</p>
              
              {patient.patient_profiles?.[0]?.assigned_doctor ? (
                <div className={styles.assignedDoctor}>
                  <p><strong>Assigned Doctor:</strong></p>
                  <p>{patient.patient_profiles[0].assigned_doctor?.user_profiles?.full_name || 'Loading...'}</p>
                </div>
              ) : (
                <div className={styles.unassigned}>
                  <p><strong>Status:</strong> Unassigned</p>
                  <button 
                    onClick={() => {
                      setSelectedPatient(patient);
                      setAssignmentMode(true);
                    }}
                    className={styles.assignBtn}
                  >
                    Assign Doctor
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderDoctorManagement = () => (
    <div className={styles.doctorManagement}>
      <h2>Doctor Management</h2>
      <div className={styles.doctorGrid}>
        {allDoctors.map(doctor => (
          <div key={doctor.id} className={styles.doctorCard}>
            <div className={styles.doctorHeader}>
              <h3>{doctor.full_name}</h3>
              <span className={styles.doctorId}>{doctor.unique_identifier}</span>
            </div>
            
            <div className={styles.doctorDetails}>
              <p><strong>License:</strong> {doctor.doctor_profiles?.[0]?.medical_license}</p>
              <p><strong>Specialization:</strong> {doctor.doctor_profiles?.[0]?.specialization}</p>
              <p><strong>Experience:</strong> {doctor.doctor_profiles?.[0]?.experience_years} years</p>
              <p><strong>Phone:</strong> {doctor.phone}</p>
              <p><strong>Patient Count:</strong> {doctor.doctor_profiles?.[0]?.patient_count || 0}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  if (isLoading && !userProfile) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading admin dashboard...</p>
      </div>
    );
  }
  
  if (!userProfile) {
    return (
      <div className={styles.loadingContainer}>
        <p>Unable to load user profile. Please try refreshing the page.</p>
      </div>
    );
  }

  const renderUserApprovalModal = () => {
    if (!showUserModal || !selectedUserForApproval) return null;

    const user = selectedUserForApproval;

    return (
      <div className={styles.modalOverlay}>
        <div className={styles.modalContent}>
          <div className={styles.modalHeader}>
            <h2>User Application Review</h2>
            <button 
              onClick={() => setShowUserModal(false)}
              className={styles.closeButton}
            >
              ×
            </button>
          </div>

          <div className={styles.modalBody}>
            {/* User Overview */}
            <div className={styles.userOverview}>
              <div className={styles.userAvatar}>
                {user.full_name?.charAt(0)?.toUpperCase()}
              </div>
              <div className={styles.userBasicInfo}>
                <h3>{user.full_name}</h3>
                <span className={`${styles.roleChip} ${styles[user.role]}`}>
                  {user.role.toUpperCase()}
                </span>
                <p className={styles.applicationDate}>
                  Applied: {new Date(user.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>

            {/* Detailed Information */}
            <div className={styles.detailSections}>
              <div className={styles.detailSection}>
                <h4>Contact Information</h4>
                <div className={styles.detailGrid}>
                  <div className={styles.detailItem}>
                    <label>Email:</label>
                    <span>{user.email}</span>
                  </div>
                  <div className={styles.detailItem}>
                    <label>Phone:</label>
                    <span>{user.phone}</span>
                  </div>
                  <div className={styles.detailItem}>
                    <label>Address:</label>
                    <span>{user.address}</span>
                  </div>
                  <div className={styles.detailItem}>
                    <label>ID:</label>
                    <span className={styles.uniqueId}>{user.unique_identifier}</span>
                  </div>
                </div>
              </div>

              {/* Role-specific information */}
              {user.role === 'patient' && user.patient_profiles?.[0] && (
                <div className={styles.detailSection}>
                  <h4>Patient Information</h4>
                  <div className={styles.detailGrid}>
                    <div className={styles.detailItem}>
                      <label>Blood Group:</label>
                      <span>{user.patient_profiles[0].blood_groups?.blood_type || 'Not specified'}</span>
                    </div>
                    <div className={styles.detailItem}>
                      <label>Emergency Contact:</label>
                      <span>{user.patient_profiles[0].emergency_contact_name || 'Not specified'}</span>
                    </div>
                    <div className={styles.detailItem}>
                      <label>Emergency Phone:</label>
                      <span>{user.patient_profiles[0].emergency_contact_phone || 'Not specified'}</span>
                    </div>
                    {user.patient_profiles[0].medical_history && (
                      <div className={styles.detailItem}>
                        <label>Medical History:</label>
                        <span>{user.patient_profiles[0].medical_history}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {user.role === 'doctor' && user.doctor_profiles?.[0] && (
                <div className={styles.detailSection}>
                  <h4>Doctor Information</h4>
                  <div className={styles.detailGrid}>
                    <div className={styles.detailItem}>
                      <label>Medical License:</label>
                      <span>{user.doctor_profiles[0].medical_license}</span>
                    </div>
                    <div className={styles.detailItem}>
                      <label>Specialization:</label>
                      <span>{user.doctor_profiles[0].specialization}</span>
                    </div>
                    <div className={styles.detailItem}>
                      <label>Experience:</label>
                      <span>{user.doctor_profiles[0].experience_years} years</span>
                    </div>
                    <div className={styles.detailItem}>
                      <label>Consultation Fee:</label>
                      <span>${user.doctor_profiles[0].consultation_fee || 'Not specified'}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className={styles.modalActions}>
            <button 
              onClick={() => handleApproveUser(user.id, user.role)}
              disabled={isLoading}
              className={styles.approveButton}
            >
              {isLoading ? 'Processing...' : 'Approve Application'}
            </button>
            <button 
              onClick={() => handleRejectUser(user.id)}
              disabled={isLoading}
              className={styles.rejectButton}
            >
              {isLoading ? 'Processing...' : 'Reject Application'}
            </button>
            <button 
              onClick={() => setShowUserModal(false)}
              className={styles.cancelButton}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      <Navbar />
      {renderUserApprovalModal()}
      <div className={styles.dashboardContainer}>
        <div className={styles.dashboardHeader}>
          <div className={styles.welcomeSection}>
            <h1>Admin Dashboard</h1>
            <p>Welcome, {userProfile?.full_name}</p>
            <p>Managing {hospitalData?.name || 'Hospital'}</p>
          </div>
        </div>

        <div className={styles.tabNavigation}>
          <button 
            className={activeTab === 'overview' ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={activeTab === 'approvals' ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab('approvals')}
          >
            Pending Approvals ({dashboardStats.pendingPatients + dashboardStats.pendingDoctors})
          </button>
          <button 
            className={activeTab === 'patients' ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab('patients')}
          >
            Patients ({dashboardStats.activePatients})
          </button>
          <button 
            className={activeTab === 'doctors' ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab('doctors')}
          >
            Doctors ({dashboardStats.activeDoctors})
          </button>
        </div>

        <div className={styles.tabContent}>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'approvals' && renderPendingApprovals()}
          {activeTab === 'patients' && renderPatientManagement()}
          {activeTab === 'doctors' && renderDoctorManagement()}
        </div>
      </div>
    </>
  );
}

export default withAuth(AdminDashboard, ['admin']);