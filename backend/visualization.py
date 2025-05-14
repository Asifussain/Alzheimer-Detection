# backend/visualization.py
# ... (imports and other functions remain the same as the last version) ...
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import scipy
from scipy.signal import welch
from scipy.integrate import simpson
import os
import io
import base64
import traceback

# --- Configuration ---
FREQ_BANDS = {"Delta": [0.5, 4], "Theta": [4, 8], "Alpha": [8, 13], "Beta": [13, 30], "Gamma": [30, 50]}
MIN_SAMPLES_FOR_WELCH = 128 # Adjust based on DEFAULT_FS and desired resolution

# --- Validation Helper --- (Keep as is)
def validate_eeg_data(eeg_data, func_name, min_samples=1):
    # ... (Keep validation logic from previous version) ...
    if eeg_data is None: print(f"Validation Error ({func_name}): Input eeg_data is None."); return False, None
    if not isinstance(eeg_data, np.ndarray): print(f"Validation Error ({func_name}): Input eeg_data is not np array."); return False, None
    if eeg_data.ndim != 2: print(f"Validation Error ({func_name}): Input eeg_data dims={eeg_data.ndim}, expected 2."); return False, None
    if eeg_data.shape[0] < min_samples: print(f"Validation Error ({func_name}): Samples={eeg_data.shape[0]}, required={min_samples}."); return False, None
    if eeg_data.shape[1] < 1: print(f"Validation Error ({func_name}): Channels={eeg_data.shape[1]}, required >= 1."); return False, None
    if not np.issubdtype(eeg_data.dtype, np.number):
        try: eeg_data = eeg_data.astype(float); print(f"Validation Info ({func_name}): Converted dtype to float.")
        except ValueError as e: print(f"Validation Error ({func_name}): Failed data conversion to float: {e}"); return False, None
    if np.isnan(eeg_data).any(): print(f"Validation Warning ({func_name}): Data contains NaNs.")
    if np.isinf(eeg_data).any(): print(f"Validation Warning ({func_name}): Data contains Infs.")
    if np.all(np.isnan(eeg_data)): print(f"Validation Error ({func_name}): Data is all NaNs."); return False, eeg_data
    return True, eeg_data

# --- Visualization Functions ---
def generate_stacked_timeseries_image(eeg_data, fs, channel_names=None):
    fig = None
    print("--- generate_stacked_timeseries_image ---")
    is_valid, eeg_data = validate_eeg_data(eeg_data, "TS Plot", min_samples=2)
    if not is_valid: return None
    if not isinstance(fs, (int, float)) or fs <= 0: print(f"Error (TS Plot): Invalid fs={fs}"); return None

    try:
        n_samples, n_channels = eeg_data.shape
        print(f"TS Plot: Generating plot data {eeg_data.shape}, fs={fs}")
        time = np.arange(n_samples) / fs
        print("TS Plot: Creating figure...")
        fig = plt.figure(figsize=(12, 8))

        # Calculate offset
        with np.errstate(invalid='ignore'): std_dev = np.nanstd(eeg_data); finite_range = np.nanmax(eeg_data) - np.nanmin(eeg_data)
        if np.isfinite(std_dev) and std_dev > 1e-9: offset_val = std_dev * 5
        elif np.isfinite(finite_range) and finite_range > 1e-9: offset_val = finite_range * 0.5
        else: offset_val = 1.0
        offsets = np.arange(n_channels) * offset_val
        print(f"TS Plot: Offset calculated ({offset_val:.3f}). Plotting channels...")

        for i in range(n_channels):
            # print(f"TS Plot: Plotting channel {i}...") # Potentially too verbose
            channel_data = eeg_data[:, i]
            label = f"Ch {i+1}" if channel_names is None or i >= len(channel_names) else channel_names[i]
            plt.plot(time, channel_data - offsets[n_channels - 1 - i], lw=0.8, label=label)

        print("TS Plot: Setting labels and layout...")
        plt.xlabel("Time (s)"); plt.ylabel("Channels (stacked)"); plt.title("Stacked EEG Time Series")
        plt.yticks([], []); plt.grid(True, axis='x', linestyle=':', alpha=0.6); plt.margins(x=0.01, y=0.01); plt.tight_layout(pad=1.5)

        print("TS Plot: Saving figure to buffer...")
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close()
        print("--- generate_stacked_timeseries_image successful ---")
        return f"data:image/png;base64,{image_base64}"
    except Exception as e: print(f"ERROR in generate_stacked_timeseries_image: {e}"); traceback.print_exc(); return None
    finally:
        if fig is not None: plt.close(fig); print("TS Plot: Figure closed.")

# --- calculate_psd --- (Keep as is from previous version)
def calculate_psd(eeg_data, fs, nperseg_mult=1):
    print("--- calculate_psd ---")
    if not isinstance(fs, (int, float)) or fs <= 0: print(f"Error (PSD Calc): Invalid fs={fs}"); return None, None
    try:
        n_samples, n_channels = eeg_data.shape
        nperseg = min(n_samples, int(fs * nperseg_mult))
        if nperseg < 2 : print(f"Error (PSD Calc): nperseg ({nperseg}) must be >= 2."); return None, None
        print(f"PSD Calc: Using nperseg={nperseg}, fs={fs}, n_samples={n_samples}, n_channels={n_channels}")
        freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, axis=0, average='mean')
        print(f"PSD Calc: Welch outputs - freqs: {freqs.shape}, psd: {psd.shape}")
        if np.isnan(psd).all(): print("Warning (PSD Calc): Welch result all NaNs.")
        if np.isinf(psd).any(): print("Warning (PSD Calc): Welch result contains Infs.")
        print("--- calculate_psd successful ---")
        return freqs, psd
    except ValueError as ve: print(f"Error (PSD Calc): Welch error: {ve}"); traceback.print_exc(); return None, None
    except Exception as e: print(f"Error (PSD Calc): Unexpected error: {e}"); traceback.print_exc(); return None, None

def generate_average_psd_image(eeg_data, fs, freq_bands=FREQ_BANDS):
    fig = None
    print("--- generate_average_psd_image ---")
    is_valid, eeg_data = validate_eeg_data(eeg_data, "PSD Plot", min_samples=MIN_SAMPLES_FOR_WELCH)
    if not is_valid: return None
    if not isinstance(fs, (int, float)) or fs <= 0: print(f"Error (PSD Plot): Invalid fs={fs}"); return None

    try:
        print("PSD Plot: Calculating PSD...")
        freqs, psd = calculate_psd(eeg_data, fs)
        if freqs is None or psd is None: print("Error (PSD Plot): PSD calculation failed."); return None
        if psd.size == 0: print("Warning (PSD Plot): PSD result empty."); return None

        print("PSD Plot: Averaging PSD...")
        with np.errstate(all='ignore'): psd_avg = np.nanmean(psd, axis=1)
        if np.all(np.isnan(psd_avg)): print("Error (PSD Plot): Average PSD is all NaNs."); return None

        print("PSD Plot: Creating figure...")
        fig = plt.figure(figsize=(8, 5)); plt.semilogy(freqs, psd_avg, lw=1.5, color='k')

        print("PSD Plot: Plotting bands...")
        for band, (low, high) in freq_bands.items():
            try:
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                valid_indices = np.logical_and(idx_band, np.isfinite(psd_avg))
                if np.any(valid_indices):
                    plt.fill_between(freqs[valid_indices], psd_avg[valid_indices], alpha=0.3, label=f"{band}")
                # else: print(f"PSD Plot: No valid data for band '{band}'") # Optional log
            except Exception as band_e: print(f"Error plotting band '{band}': {band_e}")

        print("PSD Plot: Setting labels and layout...")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Avg PSD ($\mu V^2$/Hz) [log scale]"); plt.title("Average Power Spectral Density")
        max_freq = min(fs / 2 if fs else 50, 50); plt.xlim([0, max_freq])
        valid_psd_indices = np.logical_and(freqs >= 0.5, np.isfinite(psd_avg))
        if np.any(valid_psd_indices):
             valid_psd_values = psd_avg[valid_psd_indices]
             min_psd = np.min(valid_psd_values); max_psd = np.max(valid_psd_values)
             if min_psd > 0 and max_psd > min_psd: plt.ylim([min_psd / 10, max_psd * 10])
             elif max_psd > 0: plt.ylim([max_psd / 1000, max_psd * 10])
        plt.legend(fontsize='small'); plt.grid(True, which='both', linestyle=':', alpha=0.6); plt.tight_layout(pad=1.5)

        print("PSD Plot: Saving figure to buffer...")
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close()
        print("--- generate_average_psd_image successful ---")
        return f"data:image/png;base64,{image_base64}"
    except Exception as e: print(f"ERROR in generate_average_psd_image: {e}"); traceback.print_exc(); return None
    finally:
        if fig is not None: plt.close(fig); print("PSD Plot: Figure closed.")

# --- generate_descriptive_stats --- (Keep as is from previous version)
def generate_descriptive_stats(eeg_data, fs, freq_bands=FREQ_BANDS):
    print("--- generate_descriptive_stats ---")
    stats = {}; band_powers = {band: {'absolute': None, 'relative': None} for band in freq_bands}; stats['avg_band_power'] = band_powers
    is_valid, eeg_data = validate_eeg_data(eeg_data, "Stats", min_samples=MIN_SAMPLES_FOR_WELCH)
    if not is_valid: return {'error': 'Invalid EEG data for stats'}
    if not isinstance(fs, (int, float)) or fs <= 0: return {'error': f'Invalid fs: {fs}'}
    try:
        print(f"Stats: Calculating Std Dev...")
        with np.errstate(invalid='ignore'): stats['std_dev_per_channel'] = np.nanstd(eeg_data, axis=0).tolist()
        print("Stats: Calculating PSD for band power...")
        freqs, psd = calculate_psd(eeg_data, fs)
        if freqs is None or psd is None: print("Warning (Stats): PSD failed. Skipping band power."); return stats
        with np.errstate(all='ignore'): psd_avg = np.nanmean(psd, axis=1)
        if np.all(np.isnan(psd_avg)): print("Error (Stats): Average PSD is all NaNs."); return stats
        total_power_freq_max = min(fs / 2 if fs else 50, 50); idx_total = np.logical_and(freqs >= 0.5, freqs <= total_power_freq_max)
        finite_mask_total = np.logical_and(idx_total, np.isfinite(psd_avg)); valid_psd_total = psd_avg[finite_mask_total]; valid_freqs_total = freqs[finite_mask_total]
        total_power = 0
        if valid_freqs_total.size > 1:
            try:
                total_power = simpson(valid_psd_total, x=valid_freqs_total)
            except Exception as simpson_e:
                print(f"Warning (Stats): Simpson error total power: {simpson_e}")
        if total_power <= 0: print("Warning (Stats): Total power <= 0.")
        print(f"Stats: Total power ({0.5}-{total_power_freq_max} Hz) = {total_power:.4f}")
        
        for band, (low, high) in freq_bands.items():
            try:
                abs_power = 0
                rel_power = 0
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                finite_mask_band = np.logical_and(idx_band, np.isfinite(psd_avg))
                valid_psd_band = psd_avg[finite_mask_band]
                valid_freqs_band = freqs[finite_mask_band]
                
                if valid_freqs_band.size > 1:
                    try:
                        abs_power = simpson(valid_psd_band, x=valid_freqs_band)
                        rel_power = abs_power / total_power if total_power > 0 else 0
                    except Exception as simpson_band_e:
                        print(f"Warning (Stats): Simpson error band '{band}': {simpson_band_e}")
                
                band_powers[band] = {'absolute': abs_power, 'relative': rel_power}
            except Exception as band_calc_e:
                print(f"Error calculating band '{band}': {band_calc_e}")
                band_powers[band] = {'error': str(band_calc_e)}
        stats['avg_band_power'] = band_powers; print("--- generate_descriptive_stats successful ---")
        return stats
    except Exception as e: print(f"Error calculating stats: {e}"); traceback.print_exc(); return {'error': f'Stats calc failed: {str(e)}'}