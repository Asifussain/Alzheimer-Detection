// frontend/components/FileUpload.jsx
import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import styles from '../styles/UploadSection.module.css';
// Import dynamic from next/dynamic
import dynamic from 'next/dynamic';

// Dynamically import LoadingSpinner with ssr: false
const LoadingSpinner = dynamic(() => import('./LoadingSpinner'), { ssr: false });

const BACKEND_URL = 'http://localhost:5000'; // Your backend URL

export default function FileUpload({ userId }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const router = useRouter();

  // handleFileChange, handleDrag, handleDrop remain the same...
  const handleFileChange = (event) => {
    setError(''); // Clear previous errors
    const selectedFile = event.target.files?.[0];
    console.log("File selected via change event:", selectedFile); // DEBUG LOG
    if (selectedFile) {
      if (selectedFile.name.toLowerCase().endsWith('.npy')) {
        setFile(selectedFile);
        console.log("File state set:", selectedFile.name); // DEBUG LOG
      } else {
        console.error("Invalid file type selected:", selectedFile.name); // DEBUG LOG
        setError('Invalid file type. Please upload a .npy file.');
        setFile(null);
        event.target.value = null; // Reset file input
      }
    } else {
        setFile(null);
    }
  };

   // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

   // Handle drop events
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(''); // Clear previous errors
    const droppedFile = e.dataTransfer.files?.[0];
    console.log("File selected via drop event:", droppedFile); // DEBUG LOG
    if (droppedFile) {
       if (droppedFile.name.toLowerCase().endsWith('.npy')) {
        setFile(droppedFile);
         console.log("File state set via drop:", droppedFile.name); // DEBUG LOG
      } else {
        console.error("Invalid file type dropped:", droppedFile.name); // DEBUG LOG
        setError('Invalid file type. Please upload a .npy file.');
        setFile(null);
      }
    }
  }, []);


  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(''); // Clear previous errors

    console.log("handleSubmit triggered. Current file state:", file); // DEBUG LOG

    if (!file) {
      console.error("Upload attempt with no file selected."); // DEBUG LOG
      setError('Please select a file first.');
      return;
    }
    if (!userId) {
        console.error("Upload attempt without user ID."); // DEBUG LOG
        setError('User not identified. Please log in again.');
        return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);

    try {
      console.log(`Sending POST request to ${BACKEND_URL}/api/predict`); // DEBUG LOG
      const response = await fetch(`${BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      console.log("Received response status:", response.status); // DEBUG LOG

      if (!response.ok) {
        let errorMsg = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.json();
            console.error("Backend error response:", errorData); // DEBUG LOG
            errorMsg = errorData.error || JSON.stringify(errorData) || errorMsg;
        } catch (parseError) {
            console.error("Could not parse error response as JSON:", parseError); // DEBUG LOG
            errorMsg = `${errorMsg} - ${response.statusText}`;
        }
        throw new Error(errorMsg);
      }

      const result = await response.json();
      console.log("Backend success response:", result); // DEBUG LOG

      router.push({
        pathname: '/result',
        query: {
          prediction: result.prediction,
          filename: result.filename,
          predictionId: result.prediction_id // Pass the ID
        },
      });

    } catch (err) {
      console.error('Upload failed:', err); // DEBUG LOG
      setError(`Error uploading file: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={styles.uploadContainer}>
      <form onSubmit={handleSubmit} className={styles.uploadForm} onDragEnter={handleDrag}>
        <label
          htmlFor="file-upload"
          className={`${styles.dropZone} ${dragActive ? styles.dragActive : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <p>Drag & drop your .npy file here</p>
          <p>or</p>
          <input
            type="file"
            id="file-upload"
            accept=".npy"
            onChange={handleFileChange}
            className={styles.inputFile}
            aria-label="File upload"
          />
          <label htmlFor="file-upload" className={styles.browseButton}>
            Browse File
          </label>
          {file && <p className={styles.fileName}>Selected: {file.name}</p>}
        </label>

        <button type="submit" disabled={!file || isUploading} className={styles.uploadButton}>
          {/* Use the dynamically imported LoadingSpinner */}
          {isUploading ? <LoadingSpinner /> : 'Analyze EEG'}
        </button>
        {error && <p className={styles.errorMessage}>{error}</p>}
      </form>
    </div>
  );
}