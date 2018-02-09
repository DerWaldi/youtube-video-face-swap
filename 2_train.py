import os
import sys
import argparse
import glob

import cv2
import numpy
from tqdm import tqdm

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

# check out directory directory and create if necessary
if not os.path.exists("./models/"):
    os.makedirs("./models/")

try:
    encoder.load_weights( "models/encoder.h5"   )
    decoder_A.load_weights( "models/decoder_A.h5" )
    decoder_B.load_weights( "models/decoder_B.h5" )
    print( "loaded existing model" )
except:
    # no model to load create new one
    pass

def save_model_weights():
    encoder.save_weights( "models/encoder.h5"   )
    decoder_A.save_weights( "models/decoder_A.h5" )
    decoder_B.save_weights( "models/decoder_B.h5" )
    print( "save model weights" )

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument("--src", help="Name of a Celebrity (e.g. Angela Merkel)")
    parser.add_argument("--dst", help="Name of a Person to swap the face with (e.g. Taylor Swift)")
    parser.add_argument("--epochs", default=100000, type=int, help="Number of Epochs to train")
    args = parser.parse_args()    
    
    # check out directory directory and create if necessary
    if not os.path.exists("./out/"):
        os.makedirs("./out/")
    # empty directory
    for f in glob.glob(os.path.join("./out/", "*.jpg")):
        os.remove(f)

    # load dataset images 
    images_A = get_image_paths( "data/faces/{}".format(args.src.lower().replace(" ", "_")))
    images_B = get_image_paths( "data/faces/{}".format(args.dst.lower().replace(" ", "_")))
    # map between 0 and 1 and normalize images
    images_A = load_images( images_A ) / 255.0
    images_B = load_images( images_B ) / 255.0
    images_A += images_B.mean( axis=(0,1,2) ) - images_A.mean( axis=(0,1,2) )

    # create 100 preview images during training (to not spam your disk)
    print_rate = args.epochs // 100
    if print_rate < 1:
        print_rate = 1

    # iterate epochs and train
    for epoch in tqdm(range(args.epochs)):
        # get next training batch
        batch_size = 64
        warped_A, target_A = get_training_data( images_A, batch_size )
        warped_B, target_B = get_training_data( images_B, batch_size )

        # train and calculate loss
        loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
        loss_B = autoencoder_B.train_on_batch( warped_B, target_B )

        if epoch % 100 == 0:
            # print training loss
            # print( loss_A, loss_B )

            # save model every 100 steps
            save_model_weights()
            test_A = target_A[0:14]
            test_B = target_B[0:14]

        if epoch % print_rate == 0:
            # visualize result (orginal, decoderA result, decoderB result)
            figure_A = numpy.stack([
                test_A,
                autoencoder_A.predict( test_A ),
                autoencoder_B.predict( test_A ),
                ], axis=1 )
            figure_B = numpy.stack([
                test_B,
                autoencoder_B.predict( test_B ),
                autoencoder_A.predict( test_B ),
                ], axis=1 )

            # stack images together to create a preview sheet
            figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
            figure = figure.reshape( (4,7) + figure.shape[1:] )
            figure = stack_images( figure )

            # create image and write to disk
            figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
            cv2.imwrite( "./out/" + str(epoch) + ".jpg", figure )

    # save our model after training has finished
    save_model_weights()

