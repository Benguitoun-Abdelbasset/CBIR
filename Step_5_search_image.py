import pickle
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import os
from classes import *
# ==========================================
# CONFIGURATION
# ==========================================
DB_NAME = 'image_db.sqlite'
VOC_FILE = 'vocabulary.pkl'
QUERY_PATH = 'test_images/lamp2.jpg'  # The external image you want to search for
NUM_RESULTS = 6

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_sift_features(image_path):
    im = cv2.imread(image_path)
    if im is None: return None

    # Resize massive images for speed
    max_dim = 1024
    h, w = im.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("Error: OpenCV SIFT not available.")
        return None

    kp, des = sift.detectAndCompute(gray, None)
    
    # RootSIFT Normalization
    if des is not None:
        eps = 1e-7
        des /= (des.sum(axis=1, keepdims=True) + eps) # L1
        des = np.sqrt(des)                            # Root
    
    return des

def get_filename_from_id(db_name, imid):
    con = sqlite3.connect(db_name)
    with con:
        cur = con.execute("SELECT filename FROM imlist WHERE rowid=?", (imid,))
        res = cur.fetchone()
        return res[0] if res else None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    print("--- EXTERNAL QUERY MODE ---")
    
    # 1. Validation
    if not os.path.exists(QUERY_PATH):
        print(f"Error: The file '{QUERY_PATH}' does not exist.")
        exit()

    # 2. Load Vocabulary
    print("Loading vocabulary...")
    with open(VOC_FILE, 'rb') as f:
        voc_obj = pickle.load(f)

    # 3. Initialize Searcher
    src = Searcher(DB_NAME, voc_obj)

    # 4. Process Query Image
    print(f"Extracting features from: {QUERY_PATH}")
    descriptors = extract_sift_features(QUERY_PATH)
    
    if descriptors is None:
        print("Failed to extract features (image might be empty or unreadable).")
        exit()
        
    # Project features onto vocabulary to get histogram
    print("Projecting to visual words...")
    q_hist = voc_obj.project(descriptors)
    
    # 5. Search
    print("Searching database...")
    candidates_ids = src.candidates_from_histogram(q_hist)
    top_candidates = candidates_ids[:NUM_RESULTS]
    
    print(f"Found matches. Showing top {NUM_RESULTS}...")

    # 6. Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot Query Image
    plt.subplot(1, NUM_RESULTS + 1, 1)
    try:
        plt.imshow(mpimg.imread(QUERY_PATH))
        plt.title("QUERY")
    except:
        plt.title("QUERY (Error loading)")
    plt.axis('off')

    # Plot Results
    for i, imid in enumerate(top_candidates):
        result_img_name = get_filename_from_id(DB_NAME, imid)
        
        plt.subplot(1, NUM_RESULTS + 1, i + 2)
        try:
            plt.imshow(mpimg.imread(result_img_name))
            plt.title(f"Match {i+1}")
        except Exception as e:
            print(f"Could not load result image: {result_img_name}")
            plt.title(f"Missing File")
        plt.axis('off')
        
    plt.show()