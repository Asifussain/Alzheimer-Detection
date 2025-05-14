// frontend/components/FileUploadSection.jsx
import { useState, useRef } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from './AuthProvider';
import dynamic from 'next/dynamic';
import styles from '../styles/Hero.module.css'; // Reusing Hero styles for layout consistency
// Import dashboard styles for potential card element usage if needed later
import dashStyles from '../styles/DashboardLayout.module.css';

// Dynamically import LoadingSpinner with ssr: false
const LoadingSpinner = dynamic(() => import('./LoadingSpinner'), { ssr: false });

export default function FileUploadSection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('Input EEG data (.npy)');
  // --- State for channel index selection ---
  const [selectedChannel, setSelectedChannel] = useState(1); // Default to channel 1
  // --- End New State ---
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { user, session } = useAuth();
  const router = useRouter();
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
        if (!file.name.toLowerCase().endsWith('.npy')) {
            alert('Invalid file type. Please upload a .npy file.');
            setSelectedFile(null);
            setFileName('Input EEG data (.npy)');
            if(fileInputRef.current) fileInputRef.current.value = "";
            return;
        }
      setSelectedFile(file);
      setFileName(file.name);
    } else {
      setSelectedFile(null);
      setFileName('Input EEG data (.npy)');
    }
  };

  const handleTriggerClick = () => {
    if (user && session) {
        fileInputRef.current?.click();
    } else {
        alert("Please ensure you are logged in to upload.");
    }
  };

  // --- Handler for channel selection change ---
  const handleChannelChange = (e) => {
      const value = parseInt(e.target.value, 10);
      // Validate input is between 1 and 19
      if (!isNaN(value) && value >= 1 && value <= 19) {
          setSelectedChannel(value);
      } else if (e.target.value === '') {
          // Allow clearing the input, set to empty string or handle validation on submit
          setSelectedChannel('');
      } else {
          // If value is outside range (e.g., 0 or 20), clamp it or keep previous valid
          // For simplicity, let's just keep the current valid value if out of range
          // Or you could set to min/max:
          // if (!isNaN(value)) {
          //     setSelectedChannel(Math.max(1, Math.min(19, value)));
          // }
      }
  };
  // --- End Channel Handler ---

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!user || !session) {
      alert('Login session expired or invalid. Please log in again.');
      return;
    }
    if (!selectedFile) {
      alert('Please select a .npy file first.');
      return;
    }
    // --- Validate selected channel on submit ---
    const channelNum = parseInt(selectedChannel, 10); // Parse state value
    if (isNaN(channelNum) || channelNum < 1 || channelNum > 19) {
         alert('Please select a valid channel number (1-19) for the similarity plot.');
         return;
     }
    // --- End Validation ---


    setIsSubmitting(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('user_id', user.id);
    // --- Append selected channel index (0-based) ---
    // Subtract 1 because backend/analysis expects 0-based index
    formData.append('channel_index', channelNum - 1);
    // --- End Append Channel ---


    try {
      // Ensure this URL points to your running backend API
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api/predict';
      console.log(`Sending request to: ${apiUrl}`); // Debugging

      const res = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        // Add headers if needed, e.g., for authorization if you implement backend auth checks
        // headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (!res.ok) {
        // Try to parse error message from backend
        const errorRes = await res.json().catch(() => ({ error: 'Upload failed with status: ' + res.status }));
        throw new Error(errorRes.error || `HTTP error ${res.status}`);
      }

      const data = await res.json();
      console.log("Backend Response:", data); // Debugging

      // Navigate to results page with prediction ID
      router.push({
        pathname: '/result',
        query: {
            prediction: data.prediction,
            filename: data.filename,
            prediction_id: data.prediction_id
        },
      });

    } catch (error) {
      console.error("Upload Error:", error);
      alert(`Error during analysis: ${error.message}`);
    } finally {
      // Reset state after submission attempt
      setIsSubmitting(false);
      setSelectedFile(null);
      setFileName('Input EEG data (.npy)');
      // Optionally reset channel selection or keep it
      // setSelectedChannel(1);
      if(fileInputRef.current) fileInputRef.current.value = ""; // Clear file input visually
    }
  };

  // Use styles consistent with Patient Dashboard Card for better UI integration
  return (
    <form onSubmit={handleSubmit} className={dashStyles.uploadFormContainer}> {/* Container for layout */}
        {/* Row for File Input and Analyse Button */}
        <div className={dashStyles.uploadForm}>
            {/* Hidden file input */}
            <input
              type="file"
              accept=".npy"
              onChange={handleFileChange}
              ref={fileInputRef}
              className={styles.hiddenInput} // Keep using Hero's hidden style
              required
              disabled={isSubmitting || !user || !session}
            />
            {/* File trigger */}
            <div
              className={`${dashStyles.fileInputTrigger} ${selectedFile ? dashStyles.hasFile : ''}`}
              onClick={handleTriggerClick}
              tabIndex={0}
              role="button"
              aria-label="Select EEG file"
              title={fileName} // Show full name on hover
              style={{ cursor: (!user || !session) ? 'not-allowed' : 'pointer' }}
            >
              {fileName}
            </div>
            {/* Analyse Button */}
            <button
              type="submit"
              className={dashStyles.analyseBtn}
              disabled={isSubmitting || !user || !session || !selectedFile || selectedChannel === ''}
            >
              {isSubmitting ? 'Analysing...' : 'Analyse'}
            </button>
        </div>

       {/* Row for Channel Selection Input */}
       <div className={dashStyles.channelSelector}>
           <label htmlFor="channelSelect" className={dashStyles.channelLabel}>Plot Similarity for Channel (1-19):</label>
           <input
               type="number"
               id="channelSelect"
               name="channelSelect"
               min="1"
               max="19"
               value={selectedChannel}
               onChange={handleChannelChange}
               className={dashStyles.channelInput}
               required
               disabled={isSubmitting || !user || !session}
               placeholder="1-19" // Placeholder text
           />
       </div>

      {/* Loading Indicator */}
      {isSubmitting && (
        <div className={dashStyles.loadingContainer} style={{marginTop: '1rem', justifyContent: 'center'}}>
          <LoadingSpinner />
          <p style={{color: 'var(--text-secondary)'}}>Processing your file. This may take a minute...</p>
        </div>
      )}
    </form>
  );
}