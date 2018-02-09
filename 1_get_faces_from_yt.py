import os
import sys
import argparse
import glob

import cv2
import numpy
from tqdm import tqdm
from pytube import YouTube
from moviepy.editor import *

from face_extractor import extract_faces

def extract_faces_from_video(in_filename, keyword, limit=500):
    out_dir = './data/faces/'
    dataset = keyword.lower().replace(" ", "_")
    faces_dir = os.path.join(out_dir, dataset)

    # check directory and create if necessary
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    # empty directory
    for f in glob.glob(os.path.join(faces_dir, "*.jpg")):
        os.remove(f)
            
    # open source video
    vidcap = cv2.VideoCapture(in_filename)

    # get some parameters
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    face_counter = 0
    # iterate over the frames and apply the faceswap
    for i in tqdm(range(frames_count)):
        success,image = vidcap.read()

        if success != True:
            break           
        
        # only take every 4th frame
        if i % 4 == 0:
            try:
                # extract faces and resize to 256x256 px
                facelist = extract_faces(image, 256)

                # write all face images to disk
                for j in range(len(facelist)):            
                    cv2.imwrite(os.path.join(faces_dir, "{0}_{1}.jpg".format(i, j)), facelist[j][1])

                face_counter +=1
                # has limit been reached?
                if face_counter > limit:
                    break
            except:
                print("Unexpected error:", sys.exc_info()[0])
    # release video capture and close input video file
    vidcap.release()

def download_video(url, start=0, stop=0):     
    def on_downloaded(stream, file_handle):
        # get filename of downloaded file (we can't set a output directory) and close the handle
        fn = file_handle.name
        file_handle.close()
        # load downloaded video
        clip = VideoFileClip(fn)
        
        # clip with start and stop
        if(start >= 0 and stop >= 0):
            clip = clip.subclip(start, stop)

        # store clipped video in our temporary folder
        clip.write_videofile("./temp/src_video.mp4", progress_bar=False, verbose=False)
        # remove original downloaded file
        os.remove(fn)
    
    # download youtube video from url
    yt = YouTube(url)
    # clip file after downlaod
    yt.register_on_complete_callback(on_downloaded)
    # get first "mp4" stream as explained here: https://github.com/nficano/pytube
    yt.streams.filter(subtype='mp4').first().download()

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Apply face swap on YouTube video.')
    parser.add_argument("--url", help="YouTube Video URL (e.g. https://www.youtube.com/watch?v=XnbCSboujF4)")
    parser.add_argument("--start", default=0, type=int, help="Start time in seconds (-1 for full video)")
    parser.add_argument("--stop", default=-1, type=int, help="Stop time in seconds (-1 for full video)")
    parser.add_argument("--name", help="Name of the person (e.g. \"Taylor Swift\")")
    parser.add_argument("--limit", default=500, type=int, help="Limit of Images per Dataset")
    args = parser.parse_args() 
    
    # check out directory directory and create if necessary
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")
        
    # check directory and create if necessary
    if not os.path.isdir("./data/faces/"):
        os.makedirs("./data/faces/")
    
    print("Download video with url: {}".format(args.url))
    download_video(args.url, start=args.start, stop=args.stop)
    
    print("Extracting faces from video")    
    extract_faces_from_video("./temp/src_video.mp4", args.name, args.limit)    
    
    print("\n===============================================\n")
    print("I'm done for now, you should quality check your \ngenerated datasets in \"data/faces/\"!")
    print("\n===============================================\n")