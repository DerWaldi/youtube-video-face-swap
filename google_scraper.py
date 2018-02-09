# based on the code from Rushil Srivastava
# https://github.com/rushilsrivastava/image-scrapers/blob/master/google-scrapper.py

import requests
import time
import urllib
import argparse
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pathlib import Path
from lxml.html import fromstring
import os
import sys
from fake_useragent import UserAgent

def search(keyword):
    base_url = "https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiwoLXK1qLVAhWqwFQKHYMwBs8Q_AUICigB"

    url = base_url.format(keyword.lower().replace(" ", "+"))

    # Create a browser and resize for exact pinpoints
    browser = webdriver.Chrome()
    browser.set_window_size(1024, 768)
    print("\n===============================================\n")
    print("[%] Successfully launched Chrome Browser")

    # Open the link
    browser.get(url)
    time.sleep(1)
    print("[%] Successfully opened link.")

    element = browser.find_element_by_tag_name("body")

    print("[%] Scrolling down.")
    # Scroll down
    for i in range(30):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)  # bot id protection

    browser.find_element_by_id("smb").click()
    print("[%] Successfully clicked 'Show More Button'.")

    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)  # bot id protection

    time.sleep(1)

    print("[%] Reached end of Page.")
    # Get page source and close the browser
    source = browser.page_source
    browser.close()
    print("[%] Closed Browser.")

    return source


def download_image(link, image_data, query):
    download_image.delta += 1
    # Use a random user agent header for bot id
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # Get the image link
    try:
        # Get the file name and type
        file_name = link.split("/")[-1]
        type = file_name.split(".")[-1]
        type = (type[:3]) if len(type) > 3 else type
        if type.lower() == "jpe":
            type = "jpeg"
        if type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"

        # Download the image
        print("[%] Downloading Image #{} from {}".format(download_image.delta, link))
        try:
            urllib.request.urlretrieve(link,
                                       "data/raw/{}/".format(query) + "{}.{}".format(str(download_image.delta),
                                                                                             type))
            print("[%] Downloaded File\n")
        except Exception as e:
            download_image.delta -= 1
            print("[!] Issue Downloading: {}\n[!] Error: {}".format(link, e))
    except Exception as e:
        download_image.delta -= 1
        print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))

def scrape(keyword, limit=500):
    # set stack limit
    sys.setrecursionlimit(1000000)

    # get user input and search on google
    query = keyword.lower().replace(" ", "_")

    if not os.path.isdir("data/raw/{}".format(query)):
        os.makedirs("data/raw/{}".format(query))

    source = search(keyword)

    # Parse the page source and download pics
    soup = BeautifulSoup(str(source), "html.parser")
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # Get the links and image data
    links = soup.find_all("a", class_="rg_l")
    
    # Clip Limit
    if len(links) > limit:
        links = links[0:limit]
    
    print("[%] Indexed {} Images.".format(len(links)))
    print("\n===============================================\n")
    print("[%] Getting Image Information.\n")
    images = {}
    linkcounter = 0
    image_data = None
    for a in soup.find_all("div", class_="rg_meta"):
        r = requests.get("https://www.google.com" + links[linkcounter].get("href"), headers=headers)
        title = str(fromstring(r.content).findtext(".//title"))
        link = title.split(" ")[-1]
        print("\n[%] Getting info on: {}".format(link))
        try:
            image_data = "google", query, json.loads(a.text)["pt"], json.loads(a.text)["s"], json.loads(a.text)["st"], json.loads(a.text)["ou"], json.loads(a.text)["ru"]
            images[link] = image_data
        except Exception as e:
            images[link] = image_data
            print("[!] Issue getting data: {}\n[!] Error: {}".format(image_data, e))

        linkcounter += 1
        
        if linkcounter >= limit:
            break;

    # Open i processes to download
    download_image.delta = 0
    for i, (link) in enumerate(links):
        r = requests.get("https://www.google.com" + link.get("href"), headers=headers)
        title = str(fromstring(r.content).findtext(".//title"))
        link = title.split(" ")[-1]
        try:
            download_image(link, images[link], query)
        except Exception as e:
            pass

    print("[%] Downloaded {} images.".format(download_image.delta))