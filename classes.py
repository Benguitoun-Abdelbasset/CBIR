import pickle
import sqlite3 as sqlite  # Use standard library sqlite3
import numpy as np
from scipy.cluster.vq import kmeans, vq
import cv2
import os
import glob



# ==========================================
# CLASS: INDEXER (Database Management)
# ==========================================
class Indexer(object):
    def __init__(self, db, voc):
        """ Initialize with the name of the database and a vocabulary object. """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        """ Create the database tables. """
        try:
            self.con.execute('create table imlist(filename)')
            self.con.execute('create table imwords(imid,wordid,vocname)')
            self.con.execute('create table imhistograms(imid,histogram,vocname)')
            self.con.execute('create index im_idx on imlist(filename)')
            self.con.execute('create index wordid_idx on imwords(wordid)')
            self.con.execute('create index imid_idx on imwords(imid)')
            self.con.execute('create index imidhist_idx on imhistograms(imid)')
            self.db_commit()
        except sqlite.OperationalError:
            print("Tables already exist.")

    def get_histogram(self, imname):
        """ Return the unpickled histogram for an image (by filename). """
        cursor = self.con.cursor()
        cursor.execute("""
            SELECT histogram FROM imhistograms 
            WHERE imid = (SELECT rowid FROM imlist WHERE filename = ?) 
            """, (imname,))
        
        data = cursor.fetchone()
        if data is None:
            return None
        
        # Unpickle the data immediately so it's usable
        return pickle.loads(data[0])

    def get_histogram_from_id(self, imid):
        """ Return the unpickled histogram for an image (by ID). """
        cursor = self.con.cursor()
        cursor.execute("SELECT histogram FROM imhistograms WHERE imid=?", (imid,))
        data = cursor.fetchone()
        if data is None: return None
        return pickle.loads(data[0])

    def add_to_index(self, imname, descr):
        """ Take an image with feature descriptors, 
        project on vocabulary and add to database. """
        
        if self.is_indexed(imname): 
            return
        
        #print('indexing', imname)
        
        # Get the imid
        imid = self.get_id(imname)
        
        # Get the histogram (counts of visual words)
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]
        
        # Link each word to image
        for i in range(nbr_words):
            count = imwords[i]
            if count > 0:
                # Insert the Word ID
                self.con.execute("insert into imwords(imid,wordid,vocname) values (?,?,?)", 
                                (imid, i, self.voc.name))
        
        # Store word histogram for image
        self.con.execute("insert into imhistograms(imid,histogram,vocname) values (?,?,?)", 
                        (imid, pickle.dumps(imwords), self.voc.name))

    def is_indexed(self, imname):
        """ Returns True if imname has been indexed. """
        # USE ? placeholders for safety
        im = self.con.execute("select rowid from imlist where filename=?", (imname,)).fetchone()
        return im is not None

    def get_id(self, imname):
        """ Get an entry id and add if not present. """
        cur = self.con.execute("select rowid from imlist where filename=?", (imname,))
        res = cur.fetchone()
        if res is None:
            cur = self.con.execute("insert into imlist(filename) values (?)", (imname,))
            return cur.lastrowid
        else:
            return res[0]

# ==========================================
# CLASS: RootSIFTExtractor (descriptor extraction)
# ==========================================
class RootSIFTExtractor:
    def __init__(self):
        """Initialize the SIFT detector once."""
        # SIFT is standard in OpenCV > 4.4.0
        self.sift = cv2.SIFT_create()
        self.eps = 1e-7

    def extract_features(self, img_path):
        """
        Reads an image and computes RootSIFT descriptors.
        Returns: descriptors (numpy array) or None if failed.
        """
        try:
            # Read Image in Grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                return None

            # Detect and Compute Features
            kp, des = self.sift.detectAndCompute(img, None)

            # If no features found, return None
            if des is None:
                return None

            # --- RootSIFT Normalization ---
            # Step A: L1 Normalize (divide by sum of the vector)
            des /= (des.sum(axis=1, keepdims=True) + self.eps)
            
            # Step B: Square Root (Hellinger kernel mapping)
            des = np.sqrt(des)
            
            return des

        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None

    def save_features(self, descriptors, output_path):
        """
        Saves the numpy array of descriptors to the disk.
        """
        try:
            np.save(output_path, descriptors)
            return True
        except Exception as e:
            print(f"Error saving to {output_path}: {e}")
            return False

    def process_directory(self, folder_path, ext='jpg'):
        """
        Iterates through a folder, extracts features, and saves them
        as .npy files next to the images.
        """
        # Get list of all images
        search_pattern = os.path.join(folder_path, f'*.{ext}')
        imlist = glob.glob(search_pattern)
        
        print(f"Found {len(imlist)} images in '{folder_path}'. Starting extraction...")

        count = 0
        for imname in imlist:
            # 1. Define output filename
            featfile = os.path.splitext(imname)[0] + '.npy'

            # 2. Extract
            descriptors = self.extract_features(imname)

            if descriptors is not None:
                # 3. Save
                success = self.save_features(descriptors, featfile)
                if success:
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} images...")
            else:
                print(f"Skipping {imname} (No features found or read error).")

        print(f"Done. Successfully processed {count} images.")


# ==========================================
# CLASS: SEARCHER (Retrieval Logic)
# ==========================================
class Searcher(object):
    def __init__(self, db, voc):
        self.con = sqlite.connect(db)
        self.voc = voc
        # Create a temporary indexer just to access the get_histogram_from_id method efficiently
        self.indexer = Indexer(db, voc)

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, imword):
        """ Get list of images containing imword. """
        im_ids = self.con.execute(
            "select distinct imid from imwords where wordid=?", (int(imword),)).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, query_histogram):

        # TF-IDF + L2 normalize query
        query_vec = query_histogram * self.voc.idf
        query_vec /= (np.linalg.norm(query_vec) + 1e-7)

        query_words = query_vec.nonzero()[0]

        candidate_ids = set()
        for word_id in query_words:
            candidate_ids.update(self.candidates_from_word(word_id))

        candidate_scores = []

        for imid in candidate_ids:
            cand_hist = self.indexer.get_histogram_from_id(imid)
            if cand_hist is None:
                continue

            cand_vec = cand_hist * self.voc.idf
            cand_vec /= (np.linalg.norm(cand_vec) + 1e-7)

            score = np.dot(query_vec, cand_vec)
            candidate_scores.append((imid, score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in candidate_scores]


# ==========================================
# CLASS: VOCABULARY (Machine Learning)
# ==========================================
class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.voc = None
        self.idf = None
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k=4000, subsampling=15):
        """ Train a vocabulary from features in files. """
        descr = []
        valid_files = []
        
        # 1. Load all descriptors into a list first
        print(f"Loading features from {len(featurefiles)} files...")
        for f in featurefiles:
            try:
                d = np.load(f)
                # Handle single descriptor edge case (128,) -> (1, 128)
                if d.ndim == 1: 
                    d = d[np.newaxis, :]
                    
                if len(d) > 0:
                    descr.append(d)
                    valid_files.append(f)
            except Exception as e:
                print(f"Could not load {f}: {e}")

        # 2. Stack them efficiently (vstack once is faster than vstack in loop)
        if len(descr) > 0:
            descriptors = np.vstack(descr)
        else:
            print("No features found!")
            return

        print(f"Clustering {len(descriptors)} features into {k} words...")
        
        # 3. K-Means Clustering
        self.voc, _ = kmeans(descriptors[::subsampling], k, 1)
        self.nbr_words = self.voc.shape[0]

        # 4. Project training images to calculate IDF
        nbr_images = len(valid_files)
        imwords = np.zeros((nbr_images, self.nbr_words))
        
        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])

        # 5. Calculate IDF
        nbr_occurrences = np.sum(imwords > 0, axis=0)
        self.idf = np.log((1.0 * nbr_images) / (1.0 * nbr_occurrences + 1))
        self.trainingdata = valid_files
        print("Vocabulary training complete.")

    def project(self, descriptors, threshold=0.6): # Add threshold argument
        """ Project descriptors to vocabulary words, ignoring those too far away. """
  

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.nbr_words)
        
        descriptors = np.array(descriptors, dtype=np.float32)

        if descriptors.ndim == 1:
            descriptors = descriptors[np.newaxis, :]
            
        if descriptors.shape[1] != self.voc.shape[1]:
            return np.zeros(self.nbr_words)

        imhist = np.zeros(self.nbr_words)
        
        # 1. Capture the distances (2nd return value)
        words, distances = vq(descriptors, self.voc)
        
        # 2. Filter words based on distance threshold
        # Create a boolean mask where distance is acceptable
        valid_mask = distances <= threshold
        
        # Select only the words that passed the check
        valid_words = words[valid_mask]

        # 3. Build histogram using only valid words
        for w in valid_words:
            imhist[w] += 1
            
        return imhist