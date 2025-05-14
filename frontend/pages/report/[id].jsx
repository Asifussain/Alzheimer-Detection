import { useRouter } from 'next/router';
import Navbar from '../../components/Navbar';
import ReportViewer from '../../components/ReportViewer';
import withAuth from '../../components/withAuth'; // Ensure user is logged in
import { useAuth } from '../../components/AuthProvider';
import LoadingSpinner from '../../components/LoadingSpinner';
import styles from '../../styles/PageLayout.module.css'; // Reuse page layout styles

function ReportDetailPage() {
  const router = useRouter();
  const { id: predictionId } = router.query; // Get the ID from the URL parameter
  const { profile, loading: authLoading } = useAuth(); // Get profile for role

  if (authLoading || !router.isReady) { // Wait for router and auth
      return (
           <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
              <LoadingSpinner />
           </div>
      );
  }

  if (!predictionId) {
       return (
           <>
               <Navbar />
               <div className={styles.pageContainer}>
                   <p>Invalid report ID.</p>
               </div>
           </>
       );
  }


  return (
    <>
      <Navbar />
      <div className={styles.pageContainer} style={{maxWidth: '1000px'}}> {/* Wider container? */}
         {/* Render the ReportViewer, passing the ID and role */}
         <ReportViewer predictionId={predictionId} userRole={profile?.role} />
      </div>
    </>
  );
}

// Protect this page - ensure user is logged in, no specific role required just to view
export default withAuth(ReportDetailPage);