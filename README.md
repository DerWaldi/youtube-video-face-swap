# YouTube Video Face Swap using "DeepFakes" Autoencoder-Model

## Overview

The aim of this project is to perform a face swap on a youtube video almost automatically.<br />
The only step where a human is needed is the quality in step 1.5.

### How does it work?
Siraj Raval explains that pretty good in his video:<br/>
https://www.youtube.com/watch?v=7XchCsYtYMQ&feature=youtu.be

## Installation

### My Setup

I am using a desktop PC with one GTX1060 running ubuntu server 16.04.<br />
Training the model for 100.000 epochs takes me about 30 hours. 

### Install packages from apt
```
sudo apt-get install ffmpeg x264 libx264-dev
```
#### Install xvfb for virtual screen
```
sudo apt-get install xvfb  
```
#### Install chromedriver for image scraping
```
wget https://gist.githubusercontent.com/ziadoz/3e8ab7e944d02fe872c3454d17af31a5/raw/ff10e54f562c83672f0b1958a144c4b72c070158/install.sh
sudo sh ./install.sh
```
### Install required libraries
```
pip install -r requirements.txt
```

## Usage

### Step 1: Fetch Training Data
Scrape face images of two persons from google images.
```
python3 1_get_faces.py --name="angela merkel" --limit=500
python3 1_get_faces.py --name="taylor swift" --limit=500
```
Or scrape face images from youtube videos (e.g. interviews).
```
python3 1_get_faces_from_yt.py --url="https://www.youtube.com/watch?v=XtHwjrm4sMg" --start=30 --stop=200 --name="siraj raval" --limit=500
python3 1_get_faces_from_yt.py --url="https://www.youtube.com/watch?v=kwRM4PQdheE" --start=60 --stop=179 --name="kal penn" --limit=500
```
### Step 1.5: The Human Eye

Have a look at the extracted face images in "data/faces/"!
There will appear some missextractions, just delete the images that don't fit.

### Step 2: Train Model
Train the faceswap model with the collected face images.<br/>
In this example Merkel's face will be inserted on Taylor Swift.
```
python3 2_train.py --src="angela merkel" --dst="taylor swift" --epochs=100000
```

### Pre-trained Model

You can download a pre trained model for Angela Swift [here](https://anonfile.com/Ec8a61ddbf/Angela_Swift.zip)<br/>
Just place the "models" folder next to the code directory.

### Step 3: Apply Face Swap on YouTube Video
Perform facesqp on a youtube video.<br/>
The "--start" and "--stop" parameters define in seconds where to clip the video.<br/>
Set "--gif" to "True" if you want to export the generated video as gif file. 
```
python3 3_youtube_face_swap.py --url="https://www.youtube.com/watch?v=XnbCSboujF4" --start=0 --stop=60 --gif=False
```

## Examples
Donald Trump as Nicolas Cage:<br/>
![Example GIF](https://github.com/DerWaldi/youtube-video-face-swap/blob/master/example.gif?raw=true "Example gif")
<br/>
Angela Merkel as Taylor Swift:<br/>
![Example2 GIF](https://github.com/DerWaldi/youtube-video-face-swap/blob/master/examples2.gif?raw=true "Example2 gif")<br/>
[Video with better quality](https://github.com/DerWaldi/youtube-video-face-swap/raw/master/examples2.mp4)
<br/>
![Example3 GIF](https://github.com/DerWaldi/youtube-video-face-swap/blob/master/example3.gif?raw=true "Example3 gif")<br/>
[Video with better quality](https://github.com/DerWaldi/youtube-video-face-swap/raw/master/example3.mp4)<br/>
<br/>
The first two exampels are trained with images scraped from google, that's why the swapped faces look a bit frozen. <br/>
The last one was trained using only two videos of interviews.<br/>
You can see that it can transfer facial expressions much better than the ones trained with static images.


## What's coming next?

Since I am more into audio processing, I would like to transfer the concept of face swapping on music signals.<br/>
If you have any suggestions, please let me know.

## Credits

https://github.com/deepfakes/faceswap

https://github.com/rushilsrivastava/image-scrapers

https://gist.github.com/ziadoz/3e8ab7e944d02fe872c3454d17af31a5

Special thanks goes to Siraj Raval who inspired me to this project!<br/>
https://github.com/llSourcell/deepfakes
