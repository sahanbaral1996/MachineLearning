'''
This code scraps the images from a dynamic loading website. i.e when you scroll only part that is seen in MONITOR is LOADED.
'''
import bs4 as bs
import numpy as np
from selenium import webdriver
import urllib.request
import cv2
import time

url = 'images website URL'
driver = webdriver.Chrome()
scroll_height = 500 # it should be pixel
driver.get(url)

MAX_HEIGHT = 150000 # should be pixel
filename = ''

name_for_photo_with_dir = ''

img_urls = set()

img_tag_class = '' # find it in your website
while True:
    # execute js script
    # it will scroll scroll_height at a time
    driver.execute_script("window.scrollTo(0,scroll_height)")

    # set sleep time so that contents can load should be in sec
    time.sleep(5)

    #get total scrolled Height
    height = driver.execute_script("return document.body.scrollHeight;")

    source = driver.page_source # only content seen in screen is loaded so you will have to append the data into a file from starting to end
    soup = bs.BeautifulSoup(source, 'lxml')
    for image_src in soup.findAll('img',class_='img_tag_class'):
        img_urls.add(image_src) # using a set to prevent duplicate images because when you get to second scroll some data from 1st scroll can be present

    #check for break of condition and
    if height > MAX_HEIGHT:
        # after all img src is loaded and inserted in a set
        for urls in img_urls:
            urllib.request.urlretrieve(urls,name_for_photo_with_dir)
        break;

