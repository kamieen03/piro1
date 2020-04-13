#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

def contour(img):
    # rotate so sides of rectangle are straight
    H,W = img.shape
    new_img = np.zeros((int(1.5*H),int(1.5*W)), dtype=np.uint8)
    new_img[H//4-1:H//4-1+H, W//4-1:W//4-1+W] = img
    img = new_img
    img, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _rect = cv2.minAreaRect(cnt[0])
    angle = _rect[2]
    _box = cv2.boxPoints(_rect)
    center = sum(_box)/4
    M = cv2.getRotationMatrix2D(tuple(center),90+angle,1)
    size = max(img.shape)
    img = cv2.warpAffine(img, M, (size,size))
    nz = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(nz)
    img = img[y-5:y+h+5, x-5:x+w+5] #TODO: might be out of image bound
    show(img)

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

def rot_180(img):
    h, w = img.shape
    M = cv2.getRotationMatrix2D(tuple([w//2, h//2]),180,1)
    img = cv2.warpAffine(img, M, img.shape[::-1])
    return img

def show(img):
    cv2.imshow("XD", img)
    while True:
        if cv2.waitKey(0) == 27:
            break

def characteristic(contour):
    # get indices of first nonzero pixel in each column
    return contour.argmax(0)

def binarize(img):
    img[img!=0] = 255
    return img

def compare(chars, rev_chars):
    similarity_mat = np.zeros((len(chars), len(chars)))
    norms = [np.linalg.norm(vec) for vec in chars]
    rev_norms = [np.linalg.norm(vec) for vec in rev_chars]
    for i in range(len(chars)):
        for j in range(i):
            score1 = chars[i].dot(chars[j])/(norms[i]*norms[j])
            score2 = rev_chars[i].dot(chars[j])/(rev_norms[i]*norms[j])
            similarity_mat[i][j] = max(score1, score2)
            similarity_mat[j][i] = similarity_mat[i][j]
    return similarity_mat

def parse_similarity_mat(sim_mat):
    return sim_mat.argmax(1)

def compute_accuracy(result, path):
    with open(path+'correct.txt') as f:
        correct = np.array([int(num.strip()) for num in f.readlines()])
    print(correct)
    print(result)
    acc = (correct == result).sum()/len(correct)*100
    print(acc)
    return acc

def main():
    path = f'daneA/set{sys.argv[1]}/'
    # get image names
    imgs = sorted([img for img in os.listdir(path) if img[-4:] == '.png'],
            key = lambda s: int(s.split('.')[0]))
    # read images into memory
    imgs = [cv2.imread(path+img, 0) for img in imgs]
    # get contours of images
    imgs = [contour(img) for img in imgs]
    # resize images to common size
    max_h = max([img.shape[0] for img in imgs])
    max_w = max([img.shape[1] for img in imgs])
    imgs = [cv2.resize(img, (max_w, max_h)) for img in imgs]
    imgs = [binarize(img) for img in imgs]
    # get characteristics of images
    chars = [characteristic(img) for img in imgs] 
    rev_chars = [characteristic(rot_180(img)) for img in imgs]
    similarity_mat = compare(chars, rev_chars)
    result = parse_similarity_mat(similarity_mat)
    compute_accuracy(result, path)

if __name__ == '__main__':
    main()
