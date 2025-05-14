# test_path.py
import os
import traceback # Import traceback for detailed error listing

# --- IMPORTANT: Verify this path EXACTLY matches the one in the LAST FileNotFoundError message ---
# ---            Make sure there are NO single quotes around 'Exp'                  ---
path = r'C:\Users\sathw\OneDrive\Desktop\alzheimer-detection-app\backend\SIDDHI\checkpoints\classification\ADSZ-Indep\ADformer\ADSZ-Indep_ftM_sl128_ll48_pl96_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_Exp\checkpoint.pth'

print(f"Checking path: {path}")

try:
    exists = os.path.exists(path)
    print(f"Exists: {exists}") # <-- What does this print? True or False?

    if exists:
        is_file = os.path.isfile(path)
        print(f"Is file: {is_file}")
        if is_file:
            try:
                # Try opening the file for reading to check permissions/locks
                with open(path, 'rb') as f:
                    print("Successfully opened file for reading.")
            except Exception as open_err:
                print(f"Error opening file: {open_err}")
                traceback.print_exc() # Print detailed traceback for open error
        else:
             print("Path exists, but it's a directory, not a file.")
    else:
        # If path doesn't exist, check the parent directory for clues
        parent_dir = os.path.dirname(path)
        print(f"Checking parent directory: {parent_dir}")
        parent_exists = os.path.exists(parent_dir)
        print(f"Parent exists: {parent_exists}")
        if parent_exists:
                try:
                    print(f"Files/Folders in parent '{os.path.basename(parent_dir)}': {os.listdir(parent_dir)}")
                except Exception as list_e:
                     print(f"Error listing parent dir files: {list_e}")
                     traceback.print_exc() # Print detailed traceback for list error
        else:
             print("Parent directory also does not exist.")

except Exception as e:
    print(f"An unexpected error occurred during path checking: {e}")
    traceback.print_exc() # Print detailed traceback for any other error