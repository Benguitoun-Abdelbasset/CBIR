from classes import *
import pickle
import numpy as np
import os
import glob

# ==========================================
# SCRIPT 2: INDEX IMAGES INTO DATABASE
# ==========================================
if __name__ == "__main__":
    
    # CONFIGURATION
    path = 'images/'
    vocab_file = 'vocabulary.pkl'
    db_file = 'image_db.sqlite'

    # 1. Load Vocabulary
    print("--- STEP 1: LOADING VOCABULARY ---")
    if not os.path.exists(vocab_file):
        print(f"Error: {vocab_file} not found. Please run train_vocabulary.py first.")
        exit()
        
    with open(vocab_file, 'rb') as f:
        voc = pickle.load(f)
        
    # Validation check
    if voc.voc is None:
        print("Error: Loaded vocabulary appears empty or corrupted.")
        exit()

    # 2. Prepare File Lists
    imlist = glob.glob(os.path.join(path, '*.jpg'))
    imlist.sort()
    
    featlist = []
    valid_imlist = []
    
    # Verify strict matching between images and features
    for imname in imlist:
        featname = os.path.splitext(imname)[0] + '.npy'
        if os.path.exists(featname):
            featlist.append(featname)
            valid_imlist.append(imname)
            
    nbr_images = len(valid_imlist)
    print(f"Found {nbr_images} images to index.")

    # 3. Setup Indexer
    print(f"--- STEP 2: INDEXING TO {db_file} ---")
    
    # Delete old DB if you want a fresh start (Optional)
    # if os.path.exists(db_file):
    #     os.remove(db_file)
    
    indx = Indexer(db_file, voc)
    indx.create_tables()

    # 4. Loop and Index
    for i in range(nbr_images):
        # Load the features
        descr = np.load(featlist[i])
        
        # Handle edge case where only 1 descriptor exists (shape issue)
        if descr.ndim == 1:
            descr = descr[np.newaxis, :]
            
        # Add to database
        indx.add_to_index(valid_imlist[i], descr)
        
        # Optional: Print progress
        if (i+1) % 50 == 0:
            print(f"Indexed {i+1}/{nbr_images} images...")

    # 5. Commit and Close
    indx.db_commit()
    print("Indexing complete. Database committed.")