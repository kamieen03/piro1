#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

DEBUG = False

def rotate_to_common(img, h, w):
    upper = np.count_nonzero(img[:5,:])/w
    lower = np.count_nonzero(img[-5:,:])/w
    left  = np.count_nonzero(img[:,:5])/h
    right = np.count_nonzero(img[:,-5:])/h
    minimal = min([upper, lower, left, right])
    if minimal != upper:
        if minimal == lower:
            angle = -180
        elif minimal == left:
            angle = 270
        elif minimal == right:
            angle = 90

        H,W = img.shape
        size = max(H,W)
        new_img = np.zeros((size, size), dtype=np.uint8)
        new_img[size//2-H//2:size//2-H//2+H, size//2-W//2:size//2-W//2+W] = img
        img = new_img
        H,W = img.shape

        M = cv2.getRotationMatrix2D(tuple([W//2, H//2]),angle,1)
        p = (W, H)
        img = cv2.warpAffine(img, M, p)
        nz = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(nz)
        img = img[max(0, y-3):y+h, x:x+w]
    return img

def contour(img, name):
    # rotate so sides of rectangle are straight
    H,W = img.shape
    size = max(H,W)+10
    new_img = np.zeros((size,size), dtype=np.uint8)
    new_img[size//2-H//2:size//2-H//2+H, size//2-W//2:size//2-W//2+W] = img
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
    img = cv2.Canny(img,50,150,apertureSize = 3)
    img = img[y:y+h, x:x+w]

    img = rotate_to_common(img, h, w)


    if img[:10, :].sum() > img[-10:,:].sum():
        img = img[10:,:]
    else:
        img = img[:-10,:]
    img = img[:,5:-6]
    nz = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(nz)
    img = img[y:y+h, x:x+w]
    return img

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
            score = rev_chars[i].dot(chars[j])/(rev_norms[i]*norms[j])
            similarity_mat[i][j] = score
            similarity_mat[j][i] = similarity_mat[i][j]
    return similarity_mat

def parse_similarity_mat(sim_mat):
    for row in sim_mat:
        r = list(enumerate(row))
        r = sorted(r, key = lambda i_pr: i_pr[1])[::-1]
        r = [p[0] for p in r]
        for p in r:
            print(p, end=' ')
        print()
    return sim_mat.argmax(1)

def compute_accuracy(result, path):
    with open(os.path.join(path,'correct.txt')) as f:
        correct = np.array([int(num.strip()) for num in f.readlines()])
    #print(correct)
    #print(result)
    acc = (correct == result).sum()/len(correct)*100
    #print(acc)
    return acc

def main():
    path = os.path.abspath(sys.argv[1])
    N = int(sys.argv[2])
    # get image names
    img_names = sorted([img for img in os.listdir(path) if img[-4:] == '.png'],
            key = lambda s: int(s.split('.')[0]))[:N]
    # read images into memory
    imgs = [cv2.imread(os.path.join(path,img), 0) for img in img_names]
    # get contours of images
    imgs = [contour(img, name) for (img, name) in zip(imgs, img_names)]
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
    #compute_accuracy(result, path)

if __name__ == '__main__':
    main()
