// frontend/pages/admin/dashboard.jsx
import { useState } from 'react';
import { useAuth } from '@/components/AuthProvider';
import PageLayout from '@/components/PageLayout';
import UserManagement from '@/components/admin/UserManagement'; // We will create this next
import RelationshipManagement from '@/components/admin/RelationshipManagement'; // We will create this later
import styles from '@/styles/DashboardLayout.module.css'; // We will create this CSS file

export default function AdminDashboard() {
    const { profile, logout } = useAuth();
    const [activeTab, setActiveTab] = useState('users');

    if (!profile || profile.role !== 'admin') {
        // AuthProvider should handle redirection, but this is a safeguard.
        return null;
    }

    return (
        <PageLayout>
            <div className={styles.dashboardContainer}>
                <header className={styles.header}>
                    <h1>Admin Dashboard</h1>
                    <div className={styles.userInfo}>
                        <span>{profile.full_name} ({profile.unique_identifier})</span>
                        <button onClick={logout} className={styles.logoutButton}>Log Out</button>
                    </div>
                </header>

                <nav className={styles.nav}>
                    <button 
                        onClick={() => setActiveTab('users')}
                        className={activeTab === 'users' ? styles.active : ''}
                    >
                        User Management
                    </button>
                    <button 
                        onClick={() => setActiveTab('relationships')}
                        className={activeTab === 'relationships' ? styles.active : ''}
                    >
                        Assign Patients
                    </button>
                     <button 
                        onClick={() => setActiveTab('sessions')}
                        className={activeTab === 'sessions' ? styles.active : ''}
                    >
                        View EEG Sessions
                    </button>
                </nav>

                <main className={styles.content}>
                    {activeTab === 'users' && <UserManagement />}
                    {activeTab === 'relationships' && <RelationshipManagement />}
                    {/* Add a component for viewing sessions later */}
                    {activeTab === 'sessions' && <p>EEG session monitoring will be shown here.</p>}
                </main>
            </div>
        </PageLayout>
    );
}
