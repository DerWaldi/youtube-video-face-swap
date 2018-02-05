# YouTube Video Face Swap using "DeepFakes" Autoencoder-Model

## Description

The aim of this project ist to perform a face swap on a youtube video almost automatically.<br />
The only step where a human is needed is the quality in step 1.5.

## Installation

### System Setup

I am using a desktop PC with one GTX1060 running ubuntu server 16.04.<br />
Training the model for 100.000 epochs takes me about 30 hours. 

### Install packages from apt

sudo apt-get install ffmpeg x264 libx264-dev

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

pip install -r requirements.txt

## Usage

### Step 1: Fetch Training Data
```
python3 1_get_the_data.py --src="angela merkel" --dst="taylor swift" --limit=500
```
### Step 1.5: The Human Eye

Have a look at the extracted face images in "data/faces/"!
There will appear some missextractions, just delete the images that don't fit.

### Step 2: Train Model
```
python3 2_train.py --src="angela merkel" --dst="taylor swift" --epochs=100000
```
### Step 3: Apply Face Swap on YouTube Video
```
python3 3_youtube_face_swap.py --url="https://www.youtube.com/watch?v=XnbCSboujF4" --start=0 --stop=60 --gif=False
```
![Example GIF](https://github.com/DerWaldi/youtube-video-face-swap/blob/master/example.gif?raw=true "Example gif")

## Credits

https://github.com/deepfakes/faceswap

https://github.com/rushilsrivastava/image-scrapers

https://gist.github.com/ziadoz/3e8ab7e944d02fe872c3454d17af31a5

Special thanks goes to Siraj Raval who inspired me to this project!<br/>
https://github.com/llSourcell/deepfakes
