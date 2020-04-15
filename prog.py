#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import math

DEBUG = False

def get_border_points(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=30)
    lines = lines[:, 0]

    # Find base line - it is the logest detected line 
    base_line_idx = np.argmax([calc_length(line) for line in lines])
    base_line = lines[base_line_idx]

    # Find sides - check angle between lines and distance between endpoints
    a, b = np.split(base_line, 2)
    side1 = get_side_line(a, base_line, lines)
    side2 = get_side_line(b, base_line, lines)

    return np.array([base_line, side1,side2])


def get_side_line(e, base_line, lines):
    angle_acc = 5
    side_lines = [line for line in lines \
        if (90 - angle_acc) < (abs(calc_angle(line) - calc_angle(base_line))) < (90 + angle_acc) and is_in_range(e, line)]

    return side_lines[np.argmax([calc_length(line) for line in side_lines])]


def is_in_range(e, line):
    r = 60
    square_range = math.pow(r, 2)
    return math.pow(line[0]-e[0], 2) + math.pow(line[1]-e[1], 2) <= square_range \
        or math.pow(line[2]-e[0], 2) + math.pow(line[3]-e[1], 2) <= square_range


def calc_angle(l):
    angle = math.atan2(l[1] - l[3], l[0] - l[2])
    if angle < 0:
        angle = angle + 2*np.pi
    return angle * 180/np.pi


def calc_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt(((x1-x2)**2)+((y1-y2)**2))


def rotate_to_common(img, h, w):
    upper = np.count_nonzero(img[:5,:])/w
    lower = np.count_nonzero(img[-5:,:])/w
    left  = np.count_nonzero(img[:,:5])/h
    right = np.count_nonzero(img[:,-5:])/h
    minimal = min([upper, lower, left, right])
    if DEBUG:
        print(f'  {upper:.2} \
                \n{left:.2}  {right:.2}\
            \n    {lower:.2}')
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
    print("Processing ", name)
    # rotate so sides of rectangle are straight
    preview = np.zeros(img.shape)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    l = get_border_points(edges)

    H,W = img.shape
    size = max(H,W)+10
    new_img = np.zeros((size,size), dtype=np.uint8)
    new_img[size//2-H//2:size//2-H//2+H, size//2-W//2:size//2-W//2+W] = img
    img = new_img
    img, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _rect = cv2.minAreaRect(cnt[0])
    angle = _rect[2]
    _box = cv2.boxPoints(_rect)

    #-----
    # x1, y1, x2, y2 = base_line[0]
    # cv2.line(preview, (x1, y1), (x2, y2), 255, 5)





    # print(lens)

    for x1, y1, x2, y2 in l:
            cv2.line(preview, (x1, y1), (x2, y2), 255, 5)
            # print(calc_angle(line[0]))
    show(preview)

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
    img = img[:,5:-5]
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
    # show(contour)
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
    img_names = sorted([img for img in os.listdir(path) if img[-4:] == '.png'],
            key = lambda s: int(s.split('.')[0]))
    # read images into memory
    imgs = [cv2.imread(path+img, 0) for img in img_names]
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
    compute_accuracy(result, path)

if __name__ == '__main__':
    main()
