import os
import sys
import argparse
import glob

import cv2
import numpy
from tqdm import tqdm

from google_scraper import scrape
from face_extractor import extract_faces

# Create virtual display for ubuntu server usage
from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 600))
display.start()

def preprocess_faces(keyword):
    in_dir = './data/raw/'
    out_dir = './data/faces/'
    dataset = keyword.lower().replace(" ", "_")
    faces_dir = os.path.join(out_dir, dataset)

    # check directory and create if necessary
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    # empty directory
    for f in glob.glob(os.path.join(faces_dir, "*.jpg")):
        os.remove(f)

    # loop through our previously scraped images
    files = glob.glob(os.path.join(in_dir, dataset, "*.jpg"))
    nFiles = len(files)
    counter = 1
    for i in tqdm(range(nFiles)):
        try:
            orig_image = cv2.imread(files[i])
            # extract faces and resize to 256x256 px
            facelist = extract_faces(orig_image, 256)
            
            # write all face images to disk
            for j in range(len(facelist)):            
                cv2.imwrite(os.path.join(faces_dir, "{0}_{1}.jpg".format(i, j)), facelist[j][1])
        except:
            print("Unexpected error:", sys.exc_info()[0])

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Scrapes face images from google and extracts the faces.')
    parser.add_argument("--name", help="Name of a person whos face you want to scrape (e.g. \"Angela Merkel\")")
    parser.add_argument("--limit", default=500, type=int, help="Limit of Images per Dataset")
    args = parser.parse_args()

    # check directory and create if necessary
    if not os.path.isdir("./data/raw/"):
        os.makedirs("./data/raw/")

    print("Step 1: scrape the images from google")
    scrape(args.name, int(args.limit))

    # check directory and create if necessary
    if not os.path.isdir("./data/faces/"):
        os.makedirs("./data/faces/")

    print("Step 2: extract the faces")
    preprocess_faces(args.name)
    
    print("\n===============================================\n")
    print("I'm done for now, you should quality check your \ngenerated datasets in \"data/faces/\"!")
    print("\n===============================================\n")