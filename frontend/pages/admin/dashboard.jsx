import { useState, useEffect } from 'react';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import AddUserInterface from '../../components/admin/AddUserInterface';
import EmailManagement from '../../components/admin/EmailManagement';
import supabase from '../../lib/supabaseClient';
import styles from '../../styles/DashboardLayout.module.css';

function AdminDashboard() {
  const { user, userProfile, hospitalData } = useAuth();
  // CRITICAL FIX: Persist active tab to prevent losing state on refresh
  const [activeTab, setActiveTab] = useState(() => {
    if (typeof window !== 'undefined') {
      return sessionStorage.getItem('admin_active_tab') || 'overview';
    }
    return 'overview';
  });
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardStats, setDashboardStats] = useState({
    totalUsers: 0,
    pendingPatients: 0,
    pendingDoctors: 0,
    pendingRadiologists: 0,
    activeAdmins: 0,
    activePatients: 0,
    activeDoctors: 0,
    activeRadiologists: 0,
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
  const [error, setError] = useState('');

  // CRITICAL FIX: Stable useEffect to prevent refresh issues
  useEffect(() => {
    let isMounted = true;
    let timeoutId;
    
    const initializeDashboard = async () => {
            console.log('User:', user ? 'Authenticated' : 'Not authenticated');
      console.log('UserProfile:', userProfile ? `Role: ${userProfile.role}` : 'Not loaded');
      
      // Wait for both user and userProfile to be loaded
      if (!user || !userProfile) {
                return;
      }

      if (userProfile.role !== 'admin') {
                setError('Admin access required');
        setIsLoading(false);
        return;
      }

      // Fetch hospital data if not already loaded
      if (!hospitalData && userProfile.hospital_id) {
                await fetchHospitalData();
      }

      // Add a small delay to ensure stable state before fetching data
      timeoutId = setTimeout(() => {
        if (isMounted) {
          fetchDashboardData();
        }
      }, 100);
    };

    initializeDashboard();

    // Cleanup function to prevent memory leaks and unwanted calls
    return () => {
      isMounted = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [user?.id, userProfile?.id, userProfile?.role]); // More stable dependencies

  // CRITICAL FIX: Stable tab switching with persistence
  const handleTabChange = (newTab) => {
        setActiveTab(newTab);
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('admin_active_tab', newTab);
    }
  };

  // ENTERPRISE FIX: Add visibility change handler to prevent data loss
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
                // Save current state to sessionStorage when page becomes hidden
        if (typeof window !== 'undefined') {
          sessionStorage.setItem('admin_active_tab', activeTab);
          sessionStorage.setItem('admin_dashboard_last_active', Date.now().toString());
        }
      } else {
                // Check if we need to refresh data when page becomes visible again
        const lastActive = sessionStorage.getItem('admin_dashboard_last_active');
        if (lastActive && Date.now() - parseInt(lastActive) > 300000) { // 5 minutes
                    fetchDashboardData(true);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [activeTab]);

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
                  console.log('User:', user);

      // Check if user profile is loaded
      if (!userProfile) {
                setIsLoading(false);
        return;
      }

      if (userProfile.role !== 'admin') {
                setError('Admin access required');
        setIsLoading(false);
        return;
      }

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
                setError('Authentication required. Please log in again.');
        setIsLoading(false);
        return;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      // Try APIs in order of preference
      let response;
      let apiUsed = 'unknown';
      
      try {
        console.log('Trying simple API...');
        response = await fetch('/api/admin/users-simple', {
          headers: {
            'Authorization': `Bearer ${session.access_token}`,
            'Content-Type': 'application/json'
          },
          signal: controller.signal
        });
        apiUsed = 'simple';
        console.log('Simple API response status:', response.status);
      } catch (simpleError) {
        console.warn('Simple API failed, trying complex API:', simpleError.message);
        try {
          response = await fetch('/api/admin/users', {
            headers: {
              'Authorization': `Bearer ${session.access_token}`,
              'Content-Type': 'application/json'
            },
            signal: controller.signal
          });
          apiUsed = 'complex';
          console.log('Complex API response status:', response.status);
        } catch (complexError) {
          console.warn('Complex API failed, trying demo API:', complexError.message);
          response = await fetch('/api/admin/demo-data', {
            signal: controller.signal
          });
          apiUsed = 'demo';
          console.log('Demo API response status:', response.status);
        }
      }

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
        }
        
        console.error('API Error Response:', errorData);
        
        if (response.status === 401) {
          throw new Error('Authentication failed. Please log in again.');
        } else if (response.status === 403) {
          throw new Error('Access denied. Admin privileges required.');
        } else if (response.status === 500 && errorData.message) {
          throw new Error(errorData.message);
        } else {
          throw new Error(errorData.error || `Server error: ${response.status}`);
        }
      }

      const result = await response.json();
            if (result.success && result.data) {
        const { pendingUsers, patients, doctors, stats } = result.data;
        
                console.log('- Pending users:', pendingUsers?.length || 0);
        console.log('- Patients:', patients?.length || 0); 
        console.log('- Doctors:', doctors?.length || 0);
        console.log('- Stats:', stats);
        
        setPendingUsers(pendingUsers || []);
        setAllPatients(patients || []);
        setAllDoctors(doctors || []);
        setDashboardStats(stats || {
          totalUsers: 0,
          pendingPatients: 0,
          pendingDoctors: 0,
          pendingRadiologists: 0,
          activeAdmins: 0,
          activePatients: 0,
          activeDoctors: 0,
          activeRadiologists: 0,
          unassignedPatients: 0
        });

                // Only cache real data, not demo data
        if (apiUsed !== 'demo') {
          sessionStorage.setItem(cacheKey, JSON.stringify(result.data));
          sessionStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
        }
        
        // Show message if using demo data
        if (apiUsed === 'demo') {
          console.warn('Using demo data - check your database configuration');
        }
        
      } else {
        throw new Error(`Invalid API response format from ${apiUsed} API`);
      }

    } catch (error) {
      console.error('Dashboard data fetch error:', error);
      if (error.name === 'AbortError') {
        console.warn('Request timed out');
        setError('Request timed out. Please check your network connection.');
      } else {
        console.warn('Failed to load dashboard data:', error.message);
        setError(`Failed to load dashboard data: ${error.message}`);
      }
      
      // Set empty data to prevent crashes
      setPendingUsers([]);
      setAllPatients([]);
      setAllDoctors([]);
      setDashboardStats({
        totalUsers: 0,
        pendingPatients: 0,
        pendingDoctors: 0,
        pendingRadiologists: 0,
        activeAdmins: 0,
        activePatients: 0,
        activeDoctors: 0,
        activeRadiologists: 0,
        unassignedPatients: 0
      });
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

  // Handler for when a new user is created
  const handleUserCreated = (newUserData) => {
    // Clear cache and refresh data
    const cacheKey = `admin_dashboard_${userProfile?.hospital_id}`;
    sessionStorage.removeItem(cacheKey);
    sessionStorage.removeItem(`${cacheKey}_timestamp`);
    
    // Refresh dashboard data
    fetchDashboardData(true);
  };


  const renderOverview = () => {
    if (error) {
      return (
        <div className={styles.errorSection}>
          <div className={styles.errorIcon}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12,2L13.09,8.26L22,9L17.5,13.74L18.18,22L12,17.77L5.82,22L6.5,13.74L2,9L10.91,8.26L12,2Z"/>
            </svg>
          </div>
          <h3>Dashboard Error</h3>
          <p>{error}</p>
          <div className={styles.errorActions}>
            <button onClick={() => {
              setError('');
              fetchDashboardData(true);
            }} className={styles.retryButton}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{marginRight: '8px'}}>
                <path d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/>
              </svg>
              Retry Loading Data
            </button>
          </div>
        </div>
      );
    }
    
    return (
    <div className={styles.overviewGrid}>
      <div 
        className={styles.statCard}
        onClick={() => handleTabChange('patients')}
      >
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Total Users</h3>
          <div className={styles.statNumber}>{dashboardStats.totalUsers || 0}</div>
          <p className={styles.statBreakdown}>
            {dashboardStats.activeAdmins || 0} admins, {dashboardStats.activePatients || 0} patients, {dashboardStats.activeDoctors || 0} doctors{dashboardStats.activeRadiologists ? `, ${dashboardStats.activeRadiologists} radiologists` : ''}
          </p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => handleTabChange('approvals')}
      >
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,17A1.5,1.5 0 0,1 10.5,15.5A1.5,1.5 0 0,1 12,14A1.5,1.5 0 0,1 13.5,15.5A1.5,1.5 0 0,1 12,17M12,10.5C12.83,10.5 13.5,9.83 13.5,9V7.5C13.5,6.67 12.83,6 12,6C11.17,6 10.5,6.67 10.5,7.5V9C10.5,9.83 11.17,10.5 12,10.5Z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Pending Reviews</h3>
          <div className={styles.statNumber}>{(dashboardStats.pendingPatients || 0) + (dashboardStats.pendingDoctors || 0) + (dashboardStats.pendingRadiologists || 0)}</div>
          <p className={styles.statBreakdown}>
            {dashboardStats.pendingPatients || 0} patients, {dashboardStats.pendingDoctors || 0} doctors{dashboardStats.pendingRadiologists ? `, ${dashboardStats.pendingRadiologists} radiologists` : ''} awaiting approval
          </p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => handleTabChange('patients')}
      >
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Active Accounts</h3>
          <div className={styles.statNumber}>{(dashboardStats.activeAdmins || 0) + (dashboardStats.activePatients || 0) + (dashboardStats.activeDoctors || 0) + (dashboardStats.activeRadiologists || 0)}</div>
          <p className={styles.statBreakdown}>
            Verified and operational user accounts
          </p>
        </div>
      </div>

      <div 
        className={styles.statCard}
        onClick={() => handleTabChange('assignments')}
      >
        <div className={styles.statIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19,19H5V5H19M19,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5A2,2 0 0,0 19,3M11,7H13V9H16V11H13V14H11V11H8V9H11V7Z"/>
          </svg>
        </div>
        <div className={styles.statContent}>
          <h3>Patient Assignments</h3>
          <div className={styles.statNumber}>{dashboardStats.unassignedPatients || 0}</div>
          <p className={styles.statBreakdown}>
            Patients requiring doctor assignment
          </p>
        </div>
      </div>
    </div>
    );
  };

  const renderPendingApprovals = () => {
    if (error) {
      return (
        <div className={styles.errorSection}>
          <div className={styles.errorIcon}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12,2L13.09,8.26L22,9L13.09,9.74L12,16L10.91,9.74L2,9L10.91,8.26L12,2M12,7A2,2 0 0,0 10,9A2,2 0 0,0 12,11A2,2 0 0,0 14,9A2,2 0 0,0 12,7Z"/>
            </svg>
          </div>
          <h3>Cannot Load Pending Approvals</h3>
          <p>{error}</p>
          <button onClick={() => {
            setError('');
            fetchDashboardData(true);
          }} className={styles.retryButton}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px' }}>
              <path d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/>
            </svg>
            Retry Loading Data
          </button>
        </div>
      );
    }
    
    return (
    <div className={styles.approvalSection}>
      <div className={styles.sectionHeader}>
        <h2>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
            <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z"/>
          </svg>
          User Approvals
        </h2>
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
            {isLoading ? (
              <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/>
              </svg>
            ) : (
              <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M11,16.5L6.5,12L7.91,10.59L11,13.67L16.59,8.09L18,9.5L11,16.5Z"/>
              </svg>
            )}
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
  };

  const renderPatientManagement = () => {
    if (error) {
      return (
        <div className={styles.errorSection}>
          <div className={styles.errorIcon}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12,2L13.09,8.26L22,9L13.09,9.74L12,16L10.91,9.74L2,9L10.91,8.26L12,2M12,7A2,2 0 0,0 10,9A2,2 0 0,0 12,11A2,2 0 0,0 14,9A2,2 0 0,0 12,7Z"/>
            </svg>
          </div>
          <h3>Cannot Load Patient Data</h3>
          <p>{error}</p>
          <button onClick={() => {
            setError('');
            fetchDashboardData(true);
          }} className={styles.retryButton}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px' }}>
              <path d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/>
            </svg>
            Retry Loading Data
          </button>
        </div>
      );
    }
    
    return (
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
  };

  const renderDoctorManagement = () => {
    if (error) {
      return (
        <div className={styles.errorSection}>
          <div className={styles.errorIcon}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12,2L13.09,8.26L22,9L13.09,9.74L12,16L10.91,9.74L2,9L10.91,8.26L12,2M12,7A2,2 0 0,0 10,9A2,2 0 0,0 12,11A2,2 0 0,0 14,9A2,2 0 0,0 12,7Z"/>
            </svg>
          </div>
          <h3>Cannot Load Doctor Data</h3>
          <p>{error}</p>
          <button onClick={() => {
            setError('');
            fetchDashboardData(true);
          }} className={styles.retryButton}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px' }}>
              <path d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/>
            </svg>
            Retry Loading Data
          </button>
        </div>
      );
    }
    
    return (
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
  };

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
            onClick={() => handleTabChange('overview')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M9,17H15V15H9V17M9,13H15V11H9V13M9,9H15V7H9V9M3,3V21H21V3H3M5,19V5H19V19H5Z"/>
            </svg>
            Overview
          </button>
          <button 
            className={activeTab === 'add-user' ? styles.activeTab : styles.tab}
            onClick={() => handleTabChange('add-user')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M15,14C12.33,14 7,15.33 7,18V20H23V18C23,15.33 17.67,14 15,14M6,10V7H4V10H1V12H4V15H6V12H9V10M15,12A4,4 0 0,0 19,8A4,4 0 0,0 15,4A4,4 0 0,0 11,8A4,4 0 0,0 15,12Z"/>
            </svg>
            Add User
          </button>
          <button 
            className={activeTab === 'approvals' ? styles.activeTab : styles.tab}
            onClick={() => handleTabChange('approvals')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,17A1.5,1.5 0 0,1 10.5,15.5A1.5,1.5 0 0,1 12,14A1.5,1.5 0 0,1 13.5,15.5A1.5,1.5 0 0,1 12,17M12,10.5C12.83,10.5 13.5,9.83 13.5,9V7.5C13.5,6.67 12.83,6 12,6C11.17,6 10.5,6.67 10.5,7.5V9C10.5,9.83 11.17,10.5 12,10.5Z"/>
            </svg>
            Pending Approvals ({dashboardStats.pendingPatients + dashboardStats.pendingDoctors})
          </button>
          <button 
            className={activeTab === 'patients' ? styles.activeTab : styles.tab}
            onClick={() => handleTabChange('patients')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            Patients ({dashboardStats.activePatients})
          </button>
          <button 
            className={activeTab === 'doctors' ? styles.activeTab : styles.tab}
            onClick={() => handleTabChange('doctors')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M12,2A3,3 0 0,1 15,5V11A3,3 0 0,1 12,14A3,3 0 0,1 9,11V5A3,3 0 0,1 12,2M19,18V20H5V18L7,16V14H9V15.5H15V14H17V16L19,18Z"/>
            </svg>
            Doctors ({dashboardStats.activeDoctors})
          </button>
          <button 
            className={activeTab === 'emails' ? styles.activeTab : styles.tab}
            onClick={() => handleTabChange('emails')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
              <path d="M20,8L12,13L4,8V6L12,11L20,6M20,4H4C2.89,4 2,4.89 2,6V18A2,2 0 0,0 4,20H20A2,2 0 0,0 22,18V6C22,4.89 21.1,4 20,4Z"/>
            </svg>
            Email Management
          </button>
        </div>

        <div className={styles.tabContent}>
          {/* ENTERPRISE FIX: Keep components mounted to preserve state */}
          <div style={{ display: activeTab === 'overview' ? 'block' : 'none' }}>
            {renderOverview()}
          </div>
          
          <div style={{ display: activeTab === 'add-user' ? 'block' : 'none' }}>
            <AddUserInterface onUserCreated={handleUserCreated} />
          </div>
          
          <div style={{ display: activeTab === 'approvals' ? 'block' : 'none' }}>
            {renderPendingApprovals()}
          </div>
          
          <div style={{ display: activeTab === 'patients' ? 'block' : 'none' }}>
            {renderPatientManagement()}
          </div>
          
          <div style={{ display: activeTab === 'doctors' ? 'block' : 'none' }}>
            {renderDoctorManagement()}
          </div>
          
          <div style={{ display: activeTab === 'emails' ? 'block' : 'none' }}>
            <EmailManagement />
          </div>
        </div>
      </div>
    </>
  );
}

export default withAuth(AdminDashboard, ['admin']);