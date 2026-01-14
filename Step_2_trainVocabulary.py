from classes import *
import pickle
import os
import glob

# ==========================================
# SCRIPT 1: TRAIN VISUAL VOCABULARY
# ==========================================
if __name__ == "__main__":
    
    # CONFIGURATION
    path = 'images/'
    k_words = 4000
    subsampling_rate = 10
    output_vocab_file = 'vocabulary.pkl'

    print("--- STEP 1: PREPARING DATA ---")
    
    # 1. Get list of valid feature files
    imlist = glob.glob(os.path.join(path, '*.jpg'))
    imlist.sort()
    
    featlist = []
    
    for imname in imlist:
        featname = os.path.splitext(imname)[0] + '.npy'
        if os.path.exists(featname):
            featlist.append(featname)
    
    print(f"Found {len(featlist)} feature files for training.")
    
    if len(featlist) == 0:
        print("Error: No .npy feature files found. Please run the feature extraction script first.")
        exit()

    # 2. Train Vocabulary
    print(f"--- STEP 2: TRAINING VOCABULARY (K={k_words}) ---")
    print("This may take a while depending on the number of images...")
    
    voc = Vocabulary('ukbenchtest')
    
    # Train: clusters features from the provided list
    voc.train(featlist, k=k_words, subsampling=subsampling_rate)
    
    # 3. Save to disk
    print(f"--- STEP 3: SAVING {output_vocab_file} ---")
    with open(output_vocab_file, 'wb') as f:
        pickle.dump(voc, f)
        
    print("Done. Vocabulary trained and saved.")