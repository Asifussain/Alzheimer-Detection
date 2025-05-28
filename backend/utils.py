import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Represent NaN/Inf as null in JSON
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)): # Handle numpy bool_
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def sanitize_for_helvetica(text_input):
    if not isinstance(text_input, str):
        text_input = str(text_input)
    
    # Define common non-ASCII characters and their ASCII or near-ASCII replacements
    replacements = {
        '•': '-',  # Bullet
        '◦': '-',  # White bullet
        '’': "'",  # Right single quotation mark
        '‘': "'",  # Left single quotation mark
        '“': '"',  # Left double quotation mark
        '”': '"',  # Right double quotation mark
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '…': '...',# Horizontal ellipsis
        '€': 'EUR',# Euro sign
        '£': 'GBP',# Pound sign
        # Add more replacements as needed
    }
    for uni_char, ascii_char in replacements.items():
        text_input = text_input.replace(uni_char, ascii_char)
    
    # Fallback: replace any remaining non-ASCII characters with a question mark
    return "".join(c if ord(c) < 128 else "?" for c in text_input)