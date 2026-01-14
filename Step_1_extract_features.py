from classes import *
import os

def main():
    # --- CONFIGURATION ---
    # Path to the folder containing your images
    # Ensure this folder exists and contains .jpg files
    IMAGE_FOLDER = 'images' 
    
    # Check if folder exists before running
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: The directory '{IMAGE_FOLDER}' does not exist.")
        # Optional: Create it for testing
        # os.makedirs(IMAGE_FOLDER)
        return

    # --- EXECUTION ---
    # 1. Instantiate the extractor
    extractor = RootSIFTExtractor()

    # 2. Run the batch processing
    print(f"Starting RootSIFT extraction on folder: {IMAGE_FOLDER}")
    extractor.process_directory(IMAGE_FOLDER, ext='jpg')

if __name__ == "__main__":
    main()