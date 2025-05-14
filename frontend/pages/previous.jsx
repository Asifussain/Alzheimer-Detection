import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import supabase from '../lib/supabaseClient';
import { useUser } from '../components/AuthProvider'; // Using useUser to get user object
import LoadingSpinner from '../components/LoadingSpinner';
import Link from 'next/link'; // Import Link
import styles from '../styles/PreviousUploads.module.css';

const ITEMS_PER_PAGE = 5;

export default function PreviousUploads() {
  // Use useUser() which you already have in AuthProvider context
  // It should return { user } based on your AuthProvider setup
  const { user } = useUser();
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [isFiltering, setIsFiltering] = useState(false);

  const totalPages = Math.ceil(totalCount / ITEMS_PER_PAGE);

  const fetchPredictions = async () => {
      // Ensure user object is available before fetching
      if (!user) {
        setLoading(false);
        setPredictions([]);
        setTotalCount(0);
        // Optional: Set an error or message indicating login is required
        // setError("Please log in to view history.");
        return;
      }
      // Reset error if user is now available
      setError(null);

      if (!isFiltering) setLoading(true);

      const rangeFrom = (currentPage - 1) * ITEMS_PER_PAGE;
      const rangeTo = rangeFrom + ITEMS_PER_PAGE - 1;
      try {
        let query = supabase
          .from('predictions')
          // Make sure to select 'id' for the report link
          .select('id, filename, prediction, created_at', { count: 'exact' })
          .eq('user_id', user.id); // Use user.id

        if (startDate) {
          query = query.gte('created_at', `${startDate}T00:00:00.000Z`);
        }
        if (endDate) {
          query = query.lte('created_at', `${endDate}T23:59:59.999Z`);
        }

        query = query.order('created_at', { ascending: false })
                     .range(rangeFrom, rangeTo);

        const { data, error: dbError, count } = await query;

        if (dbError) throw dbError;

        setPredictions(data || []);
        setTotalCount(count || 0);

      } catch (err) {
        console.error("Error fetching predictions:", err);
        setError("Failed to load previous predictions. Please try again later.");
        setPredictions([]);
        setTotalCount(0);
      } finally {
        setLoading(false);
        setIsFiltering(false);
      }
  };

  // Fetch predictions when user, page, or dates change
  useEffect(() => {
    fetchPredictions();
  }, [user, currentPage, startDate, endDate]); // user dependency is important

  const handleClearFilters = () => {
      setIsFiltering(true); // Indicate activity
      setStartDate('');
      setEndDate('');
      setCurrentPage(1); // Reset to first page when filters change
      // fetchPredictions will be triggered by useEffect
  };

  const handlePreviousPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  // Format timestamp helper
  const formatTimestamp = (timestamp) => {
      if (!timestamp) return 'N/A';
      try {
          return new Date(timestamp).toLocaleString(undefined, {
              year: 'numeric', month: 'short', day: 'numeric',
              hour: '2-digit', minute: '2-digit'
          });
      } catch (e) { return timestamp; } // Fallback
  };


  // --- Render Logic ---
  return (
    <>
      <Navbar />
      <div className={styles.pageContainer}>
        <h1 className={styles.pageTitle}>Previous Predictions</h1>

        {/* Main content layout */}
        <div className={styles.mainContentLayout}>

          {/* Left Column: History Table and Pagination */}
          <div className={styles.historyColumn}>
            {/* Loading State */}
            {loading && (
              <div className={`${styles.stateContainer} ${styles.loadingContainer}`}>
                <LoadingSpinner />
                <span>Loading history...</span>
              </div>
            )}

            {/* Error State */}
            {error && !loading && (
              <div className={`${styles.stateContainer} ${styles.errorContainer}`}>
                <p>{error}</p>
              </div>
            )}

             {/* No User State */}
            {!user && !loading && !error && (
                 <div className={styles.stateContainer}>
                    <p>Please log in to view your prediction history.</p>
                 </div>
            )}

            {/* Data Table (Render only if not loading, no error, user exists, and predictions have loaded) */}
            {!loading && !error && user && (
              <>
                 {predictions.length > 0 ? (
                    <>
                        <div className={styles.predictionsTableWrapper}>
                          <table className={styles.predictionsTable}>
                            <thead>
                              <tr>
                                <th>Filename</th>
                                <th>Prediction</th>
                                <th>Date Analyzed</th>
                                <th>Report</th> {/* <-- New Column Header */}
                              </tr>
                            </thead>
                            <tbody>
                              {predictions.map((p) => (
                                <tr key={p.id}>
                                  <td>{p.filename}</td>
                                  <td className={p.prediction === "Alzheimer's" ? styles.predictionCellAlzheimer : styles.predictionCellNormal}>
                                    {p.prediction}
                                  </td>
                                  <td className={styles.dateCell}>
                                    {formatTimestamp(p.created_at)}
                                  </td>
                                  {/* --- New Report Link Cell --- */}
                                  <td>
                                     <Link
                                         href={`/report/${p.id}`}
                                         className={styles.reportLinkButton} // Style this like a button
                                     >
                                         View Report
                                     </Link>
                                  </td>
                                  {/* -------------------------- */}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>

                        {/* Pagination Controls */}
                        {totalPages > 1 && (
                            <div className={styles.paginationContainer}>
                                <button
                                    onClick={handlePreviousPage}
                                    disabled={currentPage === 1 || loading || isFiltering}
                                    className={styles.paginationButton}
                                >
                                    Previous
                                </button>
                                <span className={styles.pageInfo}>
                                    Page {currentPage} of {totalPages}
                                </span>
                                <button
                                    onClick={handleNextPage}
                                    disabled={currentPage === totalPages || loading || isFiltering}
                                    className={styles.paginationButton}
                                >
                                    Next
                                </button>
                            </div>
                        )}
                    </>
                 ) : (
                     /* Empty State (User logged in, no predictions/no results after filter) */
                     <div className={styles.stateContainer}>
                        <p>{(startDate || endDate) ? "No predictions found for the selected date range." : "You haven't analyzed any files yet."}</p>
                     </div>
                 )}
              </>
            )}
          </div>
          {/* End Left Column */}


          {/* Right Column: Filters */}
          <div className={styles.filterColumn}>
            {user && ( // Only show filters if logged in
                 <div className={styles.filterContainer}>
                    <h3 style={{marginBottom:'1rem', fontWeight: 500, color:'#eee'}}>Filter by Date</h3>
                    <div className={styles.filterGroup}>
                        <label htmlFor="startDate" className={styles.filterLabel}>From:</label>
                        <input
                            type="date"
                            id="startDate"
                            className={styles.dateInput}
                            value={startDate}
                            onChange={(e) => { setStartDate(e.target.value); setCurrentPage(1); setIsFiltering(true);}}
                            max={endDate || undefined} // Prevent start date > end date
                            disabled={loading || isFiltering}
                        />
                    </div>
                    <div className={styles.filterGroup}>
                        <label htmlFor="endDate" className={styles.filterLabel}>To:</label>
                        <input
                            type="date"
                            id="endDate"
                            className={styles.dateInput}
                            value={endDate}
                            onChange={(e) => { setEndDate(e.target.value); setCurrentPage(1); setIsFiltering(true); }}
                            min={startDate || undefined} // Prevent end date < start date
                            disabled={loading || isFiltering}
                        />
                    </div>
                    {(startDate || endDate) && (
                        <button
                            onClick={handleClearFilters}
                            className={styles.filterButton}
                            disabled={loading || isFiltering}
                        >
                            {isFiltering ? 'Clearing...' : 'Clear Filters'}
                        </button>
                    )}
                 </div>
            )}
          </div>
          {/* End Right Column */}

        </div>
        {/* End Main Content Layout */}

      </div>
      {/* End Page Container */}
    </>
  );
}