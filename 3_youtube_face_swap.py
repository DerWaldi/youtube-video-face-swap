import os
import sys
import argparse
import glob

import cv2
import numpy
from tqdm import tqdm
import dlib
import face_recognition
import face_recognition_models
from pytube import YouTube
from moviepy.editor import *

from umeyama import umeyama
from face_extractor import *
from utils import get_image_paths, load_images, stack_images

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

# laod trained model
try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
except:
    # error we can't faceswap without a model
    print( "No trained model found!" )
    pass

# perform the actual face swap
def face_swap(orig_image, down_scale):
    # extract face from original
    facelist = extract_faces(orig_image, 256)
    result_image = orig_image

    # iterate through all detected faces
    for (face, resized_image) in facelist:
        range_ = numpy.linspace( 128-80, 128+80, 5 )
        mapx = numpy.broadcast_to( range_, (5,5) )
        mapy = mapx.T

        # warp image like in the training
        mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
        mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )

        src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
        dst_points = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
        mat = umeyama( src_points, dst_points, True )[0:2]

        warped_resized_image = cv2.warpAffine( resized_image, mat, (64,64) ) / 255.0

        test_images = numpy.empty( ( 1, ) + warped_resized_image.shape )
        test_images[0] = warped_resized_image

        # predict faceswap using encoder A
        figure = autoencoder_A.predict(test_images)

        new_face = numpy.clip(numpy.squeeze(figure[0]) * 255.0, 0, 255).astype('uint8')
        mat_inv = umeyama( dst_points, src_points, True )[0:2]

        # insert face into extracted face
        dest_face = blend_warp(new_face, resized_image, mat_inv)

        # create an inverse affine transform matrix to insert extracted face again
        mat = get_align_mat(face)
        mat = mat * (256 - 2 * 48)
        mat[:,2] += 48    
        mat_inv = cv2.invertAffineTransform(mat)
        # insert new face into original image
        result_image = blend_warp(dest_face, result_image, mat_inv)

    # return resulting image after downscale
    return cv2.resize(result_image, (result_image.shape[1] // down_scale, result_image.shape[0] // down_scale))

def process_video(in_filename, out_filename, keep_audio=True, down_scale=2):
    # extract audio clip from src
    if keep_audio == True:
        clip = VideoFileClip(in_filename)
        clip.audio.write_audiofile("./temp/src_audio.mp3", progress_bar=False, verbose=False)
            
    # open source video
    vidcap = cv2.VideoCapture(in_filename)

    # get some parameters from input video
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # downscale the resulting video so it looks smoother
    down_scale = 2

    # create a video writer for output
    vidwriter = cv2.VideoWriter("./temp/proc_video.avi",cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width // down_scale, height // down_scale))

    # iterate over the frames and apply the faceswap
    for i in tqdm(range(frames_count)):
        success,image = vidcap.read()

        if success != True:
            # no frames left => break
            break
            
        try:
            # process next frame
            new_image = face_swap(image, down_scale)
        except:
            # if an error occurs => take the original frame
            new_image = cv2.resize(image, (width // down_scale, height // down_scale))
        vidwriter.write(new_image)    

    # releas video capture and writer
    vidcap.release()
    vidwriter.release()
    
    # apply audio clip to generated video
    if keep_audio == True:
        video = VideoFileClip("./temp/proc_video.avi")
        video.write_videofile(out_filename, audio="./temp/src_audio.mp3", progress_bar=False, verbose=False)

def video_to_gif(in_filename, out_filename):
    # load clip in moviepy and save as gif
    clip = VideoFileClip(in_filename)
    clip.write_gif(out_filename)
    
def download_video(url, start=0, stop=0):     
    def on_downloaded(stream, file_handle):
        # get filename of downloaded file (we can't set a output directory) and close the handle
        fn = file_handle.name
        file_handle.close()
        # load downloaded video into moviepy
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
    yt.register_on_complete_callback(on_downloaded)
    yt.streams.filter(subtype='mp4').first().download()

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Apply face swap on YouTube video.')
    parser.add_argument("--url", help="YouTube Video URL (e.g. https://www.youtube.com/watch?v=XnbCSboujF4)")
    parser.add_argument("--start", default=0, type=int, help="Start time in seconds (-1 for full video)")
    parser.add_argument("--stop", default=-1, type=int, help="Stop time in seconds (-1 for full video)")
    parser.add_argument("--gif", default=False, type=bool, help="Export as gif instead of video with audio")
    args = parser.parse_args() 
    
    # check out directory directory and create if necessary
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")
    
    print("Download video with url: {}".format(args.url))
    download_video(args.url, start=args.start, stop=args.stop)
    
    print("Process video")    
    process_video("./temp/src_video.mp4", "output.mp4")
    print("Stored generated video as: output.mp4")
    
    if args.gif:
        # you want a gif, you get a gif
        video_to_gif("output.mp4", "output.gif")
        print("Stored generated gif as: output.gif")
        
    print("Finished, have fun :D")
    