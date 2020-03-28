#!/usr/bin/env python3

import cv2
import numpy as np
import os

def characteristic(img):
    # rotate so sides of rectangle are straight
    img, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _rect = cv2.minAreaRect(cnt[0])
    angle = _rect[2]
    _box = cv2.boxPoints(_rect)
    center = sum(_box)/4
    M = cv2.getRotationMatrix2D(tuple(center),90+angle,1)
    img = cv2.warpAffine(img, M, img.shape[::-1])
    nz = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(nz)
    img = img[y:y+h, x:x+w]

    # rotate to common orientation
    h, w = img.shape
    if h > w:
        M = cv2.getRotationMatrix2D(tuple([h//2, w//2]),90,1)
        p = (2*h, h//2 +w//2 + 1)
        img = cv2.warpAffine(img, M, p)
        nz = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(nz)
        img = img[y:y+h, x:x+w]

    # get contour
    img, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_cnt = np.zeros(img.shape, np.uint8)
    img_cnt = cv2.drawContours(img_cnt, cnt, -1, 255, 1)

    # cut out the straight lines and cut the picture to what is left
    if img_cnt[:10, :].sum() > img_cnt[-10:,:].sum():
        img_cnt = img_cnt[10:,:]
    else:
        img_cnt = img_cnt[:-10,:]
    img_cnt = img_cnt[:,3:-3]
    nz = cv2.findNonZero(img_cnt)
    x, y, w, h = cv2.boundingRect(nz)
    img_cnt = img_cnt[y:y+h, x:x+w]

    return img_cnt
    
def rot_180(images):
    rev = []
    for img in images:
        h, w = img.shape
        M = cv2.getRotationMatrix2D(tuple([w//2, h//2]),180,1)
        rev.append(cv2.warpAffine(img, M, img.shape[::-1]))
    return rev

def match(imgs, revs):
    res = [[0 for _ in range(len(imgs))] for _ in range(len(imgs))]
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i == j: continue
            print(i, j)
            # something is wrong here ;(
            res[i][j] = min([abs(imgs[i] - imgs[j]).mean(), abs(revs[i] - imgs[j]).mean()])
            print(imgs[i].mean(), imgs[j].mean())
            cv2.imshow('i', imgs[i])
            cv2.imshow('j', imgs[j])
            cv2.imshow('i-j', imgs[i]-imgs[j])
            cv2.imshow('ri-j', revs[i] - imgs[j])
            while True:
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break
    res = np.array(res)
    print(res)



#TODO: checkout https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect#matchshapes
def main():
    path = 'daneA/set0/'
    imgs = sorted([img for img in os.listdir(path) if img[-4:] == '.png'],
            key = lambda s: int(s.split('.')[0]))
    imgs = [cv2.imread(path+img, 0) for img in imgs]
    imgs = [characteristic(img) for img in imgs] 
    max_h = max([img.shape[0] for img in imgs])
    max_w = max([img.shape[1] for img in imgs])
    imgs = [cv2.resize(img, (max_w, max_h)) for img in imgs]
    revs = rot_180(imgs)
    match(imgs, revs)
    #moments = [cv2.moments(img) for img in imgs] 
    #hum = [cv2.HuMoments(m) for m in moments]
    #hum = np.log(np.sign(hum) * hum)
    #match1(hum)


if __name__ == '__main__':
    main()
