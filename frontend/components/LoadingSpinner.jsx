// frontend/components/LoadingSpinner.jsx
import React from 'react';
import styles from '../styles/CardioLoader.module.css'; // Import the new CSS module

export default function LoadingSpinner() {
  // You can optionally accept props to override CSS variables if needed later
  // e.g., <LoadingSpinner color="#ff0000" size="60px" />
  // and apply them via inline style: style={{ '--uib-color': color, '--uib-size': size }}

  return (
    <svg
      className={styles.container} // Apply the container style
      x="0px"
      y="0px"
      viewBox="0 0 50 31.25"
      // Removed fixed height/width, size controlled by CSS
      preserveAspectRatio='xMidYMid meet'
    >
      {/* Track path (background line) */}
      <path
        className={styles.track} // Apply track style
        strokeWidth="4"
        fill="none"
        // pathLength="100" // Not needed for the static track
        d="M0.625 21.5 h10.25 l3.75 -5.875 l7.375 15 l9.75 -30 l7.375 20.875 v0 h10.25"
      />
      {/* Car path (animated line) */}
      <path
        className={styles.car} // Apply car (animated) style
        strokeWidth="4"
        fill="none"
        pathLength="100" // Important for stroke-dasharray animation
        d="M0.625 21.5 h10.25 l3.75 -5.875 l7.375 15 l9.75 -30 l7.375 20.875 v0 h10.25"
      />
    </svg>
  );
}
