// frontend/pages/doctor/dashboard.jsx
import { useState } from 'react';
import { useAuth } from '@/components/AuthProvider';
import PageLayout from '@/components/PageLayout';
import EegUpload from '@/components/doctor/EegUpload'; // To be created
import PatientList from '@/components/doctor/PatientList'; // To be created
import SessionHistory from '@/components/doctor/SessionHistory'; // To be created
import styles from '@/styles/DashboardLayout.module.css';

export default function DoctorDashboard() {
    const { profile, logout } = useAuth();
    const [activeTab, setActiveTab] = useState('upload');

    if (!profile || profile.role !== 'doctor') {
        return null; // Safeguard, AuthProvider handles redirection
    }

    return (
        <PageLayout>
            <div className={styles.dashboardContainer}>
                <header className={styles.header}>
                    <h1>Doctor Dashboard</h1>
                    <div className={styles.userInfo}>
                        <span>Dr. {profile.full_name} ({profile.unique_identifier})</span>
                        <button onClick={logout} className={styles.logoutButton}>Log Out</button>
                    </div>
                </header>

                <nav className={styles.nav}>
                    <button 
                        onClick={() => setActiveTab('upload')}
                        className={activeTab === 'upload' ? styles.active : ''}
                    >
                        New EEG Session
                    </button>
                    <button 
                        onClick={() => setActiveTab('patients')}
                        className={activeTab === 'patients' ? styles.active : ''}
                    >
                        My Patients
                    </button>
                     <button 
                        onClick={() => setActiveTab('history')}
                        className={activeTab === 'history' ? styles.active : ''}
                    >
                        Session History
                    </button>
                </nav>

                <main className={styles.content}>
                    {activeTab === 'upload' && <EegUpload />}
                    {activeTab === 'patients' && <PatientList />}
                    {activeTab === 'history' && <SessionHistory />}
                </main>
            </div>
        </PageLayout>
    );
}
