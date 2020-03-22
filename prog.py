#!/usr/bin/env python3

import cv2
import numpy as np

def characteristic(img):
    img, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # get bounding box of contour, it's center and angle of rotation
    _rect = cv2.minAreaRect(cnt[0])
    angle = _rect[2]
    _box = cv2.boxPoints(_rect)
    center = sum(_box)/4
    _box = np.int0(_box)
    img_rect = np.zeros(img.shape, np.uint8)
    img_rect = cv2.drawContours(img_rect,[_box],-1,255,6)

    # draw the contour
    img_cnt = np.zeros(img.shape, np.uint8)
    img_cnt = cv2.drawContours(img_cnt, cnt, -1, 255, 1)

    # get characteristic (nonrectangular part of the original contour)
    diff = img_cnt & (~img_rect)
    M = cv2.getRotationMatrix2D(tuple(center),90+angle,1)
    diff = cv2.warpAffine(diff, M, diff.shape[::-1])
    nz = cv2.findNonZero(diff)
    x, y, w, h = cv2.boundingRect(nz)
    diff = diff[y:y+h, x:x+w]

    return diff
    
#TODO: checkout https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect#matchshapes
def main():
    imga = cv2.imread('daneA/set0/1.png', 0)
    characteristic(imga) 

if __name__ == '__main__':
    main()
