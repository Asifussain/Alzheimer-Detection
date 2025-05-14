// frontend/components/ReportViewer.jsx
import React, { useState, useEffect, useRef } from 'react';
import supabase from '../lib/supabaseClient'; // Your Supabase JS client
import LoadingSpinner from './LoadingSpinner';
import styles from '../styles/ReportViewer.module.css'; // Assuming styles are defined here
// Import React Icons (install if needed: npm install react-icons)
import {FiShield, FiActivity, FiCheckCircle, FiAlertTriangle, FiBarChart2, FiTarget, FiZap, FiCompass, FiHash } from 'react-icons/fi';


// Helper to format metric values (e.g., percentages)
const formatMetric = (value, type = 'float') => {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    if (type === 'percent') {
        return `${(value * 100).toFixed(1)}%`;
    }
    if (type === 'float') {
        return value.toFixed(3);
    }
    return value; // Return as is for integers (counts)
};

// Metric Item Component for better structure
const MetricItem = ({ icon, label, value, unit = '', description = '' }) => (
    <div className={styles.metricItem}>
        <div className={styles.metricIconLabel}>
            {icon && React.createElement(icon, { className: styles.metricIcon })}
            <span className={styles.metricLabel}>{label}</span>
        </div>
        <span className={styles.metricValue}>{value}{unit}</span>
        {description && <span className={styles.metricDescription}>{description}</span>}
    </div>
);


export default function ReportViewer({ predictionId, userRole }) {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const isMounted = useRef(false);

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; };
  }, []);

  useEffect(() => {
    if (isMounted.current) { setReportData(null); setLoading(true); setError(null); }
    else { return; }

    if (!predictionId) { if (isMounted.current) { setLoading(false); setError("No prediction ID provided."); } return; }

    let currentFetchId = predictionId;

    const fetchReportRecord = async () => {
        try {
            console.log(`Fetching full report record from DB for ID: ${predictionId}`);
            // Select all needed columns, including consistency_metrics
            const { data, error: dbError } = await supabase
                .from('predictions')
                .select(`*`)
                .eq('id', predictionId)
                .maybeSingle();

            if (!isMounted.current || currentFetchId !== predictionId) { console.log("ReportViewer fetch aborted: component unmounted or ID changed."); return; }
            if (dbError) throw new Error(`DB fetch failed: ${dbError.message}`);
            if (!data) { setError(`Report record not found for ID: ${predictionId}. It might still be generating.`); setLoading(false); return; }

            if (!data.pdf_report_url) { console.warn("PDF URL missing."); }
            if (!data.similarity_plot_url) { console.warn("Similarity Plot URL missing."); }
            if (!data.consistency_metrics) { console.warn("Consistency Metrics missing."); }

            setReportData(data); console.log("Fetched Report Data:", data); setError(null);

        } catch (err) {
            console.error("Error fetching report record:", err);
            if (isMounted.current && currentFetchId === predictionId) { setError(err.message); }
        } finally {
            if (isMounted.current && currentFetchId === predictionId) { setLoading(false); }
        }
    };
    fetchReportRecord();
  }, [predictionId]);

  const formatProbs = (probs) => { /* Keep as before */ };

  if (loading) return <div className={styles.loadingContainer}><LoadingSpinner /> <p>Loading Report...</p></div>;
  if (error) return <div className={styles.errorContainer}>Error loading report: {error}</div>;
  if (!reportData) return <div className={styles.container}>Could not load report data.</div>;

  // Safely access data
  const stats = reportData.stats_data || {}; const avgBandPower = stats.avg_band_power || {}; const stdDevs = stats.std_dev_per_channel || [];
  const similarityResults = reportData.similarity_results || {}; const similarityInterpretation = similarityResults.interpretation || "Similarity analysis not available."; const similarityPlotUrl = reportData.similarity_plot_url; const plottedChannelIndex = similarityResults.plotted_channel_index
  // --- NEW: Safely access consistency metrics ---
  const consistencyMetrics = reportData.consistency_metrics || {};
  const numTrials = consistencyMetrics.num_trials;
  const showMetrics = numTrials > 1 && !consistencyMetrics.error && !consistencyMetrics.message;
  // --- End New ---

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Analysis Report</h2>
      <div className={styles.downloadButtonContainer}> {/* Download Button */}
        {reportData.pdf_report_url ? (<a href={reportData.pdf_report_url} download={`report_${reportData.filename || predictionId}.pdf`} className={styles.downloadButton} target="_blank" rel="noopener noreferrer">Download PDF</a>) : (<button className={styles.downloadButton} disabled>PDF N/A</button>)}
      </div>

      {/* --- Info Grid --- */}
      <div className={styles.infoGrid}>
        <div><strong>Filename:</strong> {reportData.filename || 'N/A'}</div>
        <div><strong>Analyzed:</strong> {reportData.created_at ? new Date(reportData.created_at).toLocaleString() : 'N/A'}</div>
        <div><strong>ML Prediction:</strong> <span className={reportData.prediction === "Alzheimer's" ? styles.predictionAlz : styles.predictionNorm}>{reportData.prediction || 'N/A'}</span></div>
        <div><strong>ML Confidence:</strong> {formatProbs(reportData.probabilities)}</div>
      </div>

      {/* --- Section: NEW Internal Consistency Metrics --- */}
      <div className={styles.section}>
         <h3 className={styles.sectionTitle}><FiActivity /> Internal Consistency Metrics</h3>
         {showMetrics ? (
            <>
                <p className={styles.metricsDisclaimer}>
                    These metrics measure prediction consistency across {numTrials} segments within this sample,
                    using the overall prediction ('{reportData.prediction}') as the reference. They do not reflect performance against external ground truth.
                </p>
                <div className={styles.metricsGrid}>
                     <MetricItem icon={FiTarget} label="Accuracy" value={formatMetric(consistencyMetrics.accuracy, 'percent')} description="Overall correct segment predictions" />
                     <MetricItem icon={FiCheckCircle} label="Precision (Alz)" value={formatMetric(consistencyMetrics.precision)} description="Of segments predicted Alz, % correct" />
                     <MetricItem icon={FiZap} label="Sensitivity (Recall, Alz)" value={formatMetric(consistencyMetrics.recall_sensitivity)} description="% of Alz-like segments correctly identified" />
                     <MetricItem icon={FiShield} label="Specificity (Normal)" value={formatMetric(consistencyMetrics.specificity)} description="% of Normal-like segments correctly identified" />
                     <MetricItem icon={FiBarChart2} label="F1-Score (Alz)" value={formatMetric(consistencyMetrics.f1_score)} description="Balance of Precision & Recall" />
                     <MetricItem icon={FiHash} label="Trials Analyzed" value={numTrials} description="Number of segments processed" />
                </div>
                <div className={styles.confusionMatrix}>
                     <span>Confusion (Ref: '{reportData.prediction}'):</span>
                     <span>TP: {consistencyMetrics.true_positives ?? '?'}</span>|
                     <span>TN: {consistencyMetrics.true_negatives ?? '?'}</span>|
                     <span>FP: {consistencyMetrics.false_positives ?? '?'}</span>|
                     <span>FN: {consistencyMetrics.false_negatives ?? '?'}</span>
                 </div>
            </>
         ) : consistencyMetrics.message ? (
            <p className={styles.loadingTextSmall}>({consistencyMetrics.message})</p>
         ) : consistencyMetrics.error ? (
             <p className={styles.errorTextSmall}>(Error: {consistencyMetrics.error})</p>
         ) : (
            <p className={styles.loadingTextSmall}>(Metrics not applicable or not calculated)</p>
         )}
      </div>
      {/* --- End NEW Section --- */}


      {/* --- Section: Similarity Analysis --- */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}><FiCompass/> Signal Shape Similarity (DTW)</h3>
        {similarityResults.error ? (<p className={styles.errorTextSmall}>(Error: {similarityResults.error})</p>) : (<>
            <div className={styles.interpretationBlock}> <pre className={styles.interpretationText}> {similarityInterpretation} </pre> </div>
            <div className={styles.plotContainer}>
                <h4>Channel {plottedChannelIndex !== undefined ? plottedChannelIndex + 1 : '?'} Comparison Plot</h4>
                {similarityPlotUrl ? (<img src={similarityPlotUrl} alt={`Signal Shape Comparison Plot for Channel ${plottedChannelIndex !== undefined ? plottedChannelIndex + 1 : '?'}`} className={styles.plotImage}/>) : (<p className={styles.loadingTextSmall}>(Plot not generated or URL missing)</p>)}
            </div>
        </>)}
      </div>

      {/* --- Section: Descriptive Statistics --- */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}><FiBarChart2 /> Descriptive Statistics</h3>
        {stats.error ? (<p className={styles.errorTextSmall}>(Error: {stats.error})</p>) : Object.keys(avgBandPower).length > 0 ? (
          <div className={styles.statsBlock}>
            <h4>Avg. Relative Band Power (%):</h4>
            <ul> {Object.entries(avgBandPower).map(([band, power]) => (<li key={band}><strong>{band}:</strong> {(power?.relative * 100 || 0).toFixed(2)}%</li> ))} </ul>
          </div> ) : (<p className={styles.loadingTextSmall}>Band power statistics not available.</p>)}
        {stdDevs.length > 0 && userRole !== 'patient' && (<div className={styles.statsBlock}> <h4>Std Dev per Channel (µV):</h4> <p className={styles.smallText}>{stdDevs.map(v => v?.toFixed(2) ?? 'N/A').join(', ')}</p> </div> )}
      </div>

      {/* --- Section: Standard Visualizations --- */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}><FiActivity/> Standard Visualizations</h3>
        <div className={styles.plotContainer}> <h4>Stacked Time Series</h4> {reportData.timeseries_plot_url ? (<img src={reportData.timeseries_plot_url} alt="Stacked EEG Time Series Plot" className={styles.plotImage}/>) : ( <p className={styles.loadingTextSmall}>(Plot not generated or URL missing)</p> )} </div>
        <div className={styles.plotContainer}> <h4>Average Power Spectral Density</h4> {reportData.psd_plot_url ? (<img src={reportData.psd_plot_url} alt="Average PSD Plot" className={styles.plotImage}/>) : ( <p className={styles.loadingTextSmall}>(Plot not generated or URL missing)</p> )} </div>
        {userRole === 'clinician' && (<p style={{marginTop: '1rem', color: '#aaa', textAlign: 'center'}}>(Additional Clinician-specific views could go here)</p>)}
      </div>
    </div>
  );
}