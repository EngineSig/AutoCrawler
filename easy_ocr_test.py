import easyocr
import argparse
import cv2
import numpy
import torch
import os

if __name__ == '__main__':

    # img_path = "C:/Users/SJ/Documents/GitHub/USC Course/AutoCrawler/download/재무제표 예시 이건 뭐지/"
    img_path = "C:/Users/SJ/Documents/GitHub/USC Course/AutoCrawler/"

    lang = ['ko', 'en']
    reader = easyocr.Reader(lang_list=["en"])

    for entry in os.listdir(img_path):
        if entry.endswith(".jpg") or entry.endswith(".jpg"):
            image = cv2.imread(img_path+entry)
            results = reader.detect(image)

            if len(results[0]) == 0 and len(results[1]) == 0:
                os.remove(img_path + entry)
