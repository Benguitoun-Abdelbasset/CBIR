
# Content-Based Image Retrieval (CBIR) System

This project implements a **Bag of Visual Words (BoVW)** model for image retrieval. It uses SIFT descriptors, K-Means clustering for vocabulary generation, and an inverted index with cosine similarity to find visually similar images in a dataset.

The system is designed to be modular, allowing for separate feature extraction, vocabulary training, and indexing phases.

## 📋 Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install opencv-python numpy scipy matplotlib
Note: opencv-python must be version 4.4.0 or higher to access SIFT (which is no longer patent-protected).
```

## ⚙️ Setup
Crucial Step: Before running any scripts, you must provide the image dataset.

Create a folder named images in the root directory of this project.

Place your dataset images inside this folder.

Ensure the images are in .jpg format.

Your directory structure should look like this:
```text

.
├── classes.py
├── Step_1_extract_features.py
├── Step_2_trainVocabulary.py
├── Step_3_indexImages.py
├── Step_4_benchmark.py
├── Step_5_search_image.py
├── .gitignore
└── images/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```
## 🚀 Usage Pipeline
Run the scripts in the following numerical order to build the system.

Step 1: Feature Extraction
Extracts SIFT descriptors from all .jpg images in the images/ folder.

```Bash

python Step_1_extract_features.py
Output: Generates a corresponding .npy file for each image (e.g., images/image001.npy) containing the raw descriptors.
```
Step 2: Train Vocabulary
Uses K-Means clustering to generate a "Visual Vocabulary" from the extracted features.

```Bash

python Step_2_trainVocabulary.py
Output: Creates vocabulary.pkl. This file contains the cluster centers (visual words).
```
Step 3: Index Images
Maps every image's features to the visual words (creating histograms) and stores them in a database.

```Bash

python Step_3_indexImages.py
Output: Creates image_db.sqlite. This database contains the indexed histograms for fast retrieval.
```
## 🔍 Searching & Evaluation
Once the database is built (Steps 1-3), you can use the following scripts:

Benchmark (Optional)
Evaluates the retrieval quality of the system using the UKBench scoring method (checking if top results belong to the same object group).

```Bash

python Step_4_benchmark.py
Output: Prints the average accuracy score (0.00 - 4.00) to the console.
```
Search for an Image
To search for similar images using an external query image:

Open Step_5_search_image.py.

Edit the QUERY_PATH variable to point to your query image (e.g., test_image.jpg).

Run the script:

```Bash

python Step_5_search_image.py
Output: Displays a Matplotlib window showing the query image and the top 6 matches found in the database.
```
## 📂 File Description
classes.py: Contains the core logic and classes (RootSIFTExtractor, Vocabulary, Indexer, Searcher) used by the scripts.

vocabulary.pkl: The serialized visual vocabulary model (generated).

image_db.sqlite: The SQLite database storing image histograms (generated).

.gitignore: Specifies files to be ignored by Git (e.g., huge datasets, compiled python files).
