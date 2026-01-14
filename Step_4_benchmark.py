import pickle
import sqlite3
import numpy as np
import os
from classes import *
# ==========================================
# CONFIGURATION
# ==========================================
DB_NAME = 'image_db.sqlite'
VOC_FILE = 'vocabulary.pkl'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_imlist(db_name):
    """ Returns a list of all filenames in the database in order. """
    con = sqlite3.connect(db_name)
    with con:
        cur = con.execute("SELECT filename FROM imlist ORDER BY rowid")
        return [row[0] for row in cur.fetchall()]

def compute_ukbench_score(src, indx, imlist):
    """
    Returns the average number of correct images in the top 4 results.
    Evaluates only every 10th image to save time.
    Assumes UKBench structure: images are in groups of 4 (0-3, 4-7, etc).
    """
    nbr_images = len(imlist)
    scores = []

    print(f"Benchmarking on every 10th image out of {nbr_images} images...")

    for i in range(0, nbr_images, 10):
        filename = imlist[i]

        # 1. Get the histogram for the current image (already indexed)
        h = indx.get_histogram(filename)
        if h is None:
            continue

        # 2. Query the database
        candidates = src.candidates_from_histogram(h)

        # 3. Take top 4 results
        top_4 = candidates[:4]

        # 4. Check correctness
        current_score = 0
        target_group = i // 4

        for result_id in top_4:
            # SQLite IDs are 1-based, convert to 0-based index
            result_index = result_id - 1
            result_group = result_index // 4

            if result_group == target_group:
                current_score += 1

        scores.append(current_score)

        if len(scores) % 10 == 0:
            print(
                f"Processed {len(scores)} query images "
                f"(up to index {i})... Current Avg Score: {np.mean(scores):.2f}"
            )

    return np.mean(scores)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Vocabulary
    print(f"Loading vocabulary from {VOC_FILE}...")
    with open(VOC_FILE, 'rb') as f:
        voc_obj = pickle.load(f)

    # 2. Initialize Core Classes
    src = Searcher(DB_NAME, voc_obj)
    indx = Indexer(DB_NAME, voc_obj) 

    print("--- BENCHMARK MODE ---")
    
    # 3. Get list of all images
    imlist = get_imlist(DB_NAME)
    
    if len(imlist) == 0:
        print("Error: Database is empty. Please index images first.")
        exit()
        
    # 4. Run the scoring
    score = compute_ukbench_score(src, indx, imlist)
    
    print("-" * 30)
    print(f"FINAL UKBENCH SCORE: {score:.2f} / 4.00")
    print("-" * 30)
    print("Note: A score of 4.0 means every image found its 3 siblings perfectly.")