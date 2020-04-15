#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import math
from sklearn.cluster import DBSCAN

DEBUG = False

def cluster_lines(lines):
    db = DBSCAN(eps=50, min_samples=1).fit(lines)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    new_lines = []
    used_labs = []
    for line, lab in zip(lines, labels):
        if lab not in used_labs:
            used_labs.append(lab)
            new_lines.append(line)
    return new_lines

def good_line(img,x1,x2,y1,y2):
    H,W = img.shape
    valid = lambda x,y: x >= 0 and x < W and y >= 0 and y < H
    SAMPLES = 500

    dir = np.array([x2-x1, y2-y1])
    dir = dir / SAMPLES
    normal = np.array([y2-y1, -(x2-x1)])
    normal = normal / np.linalg.norm(normal)

    base = np.array([x1, y1])
    left, right = 0, 0
    for i in range(SAMPLES):
        base = base + dir
        if not valid(base[0],base[1]):
            continue
        for j in range(5,8):
            x,y = base+j*normal
            if not valid(x,y): continue
            x,y = int(x),int(y)
            left += img[y,x]
            x,y = base-j*normal
            if not valid(x,y): continue
            x,y = int(x),int(y)
            right += img[y,x]
    if left == 0 or right == 0:
        return True, max(left,right)
    if left/right > 100 or right/left > 100:
        return True, max(left,right)
    return False, 0

def intersection(line1, line2):
    x1,x2,y1,y2 = line1
    x3,x4,y3,y4 = line2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom) < 0.001:
        return -1, -1
    x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/denom
    y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/denom
    x, y = int(x), int(y)
    return x, y

def three_line(lines_blens, img):
    '''
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    H,W = img.shape
    valid = lambda x,y: x >= 0 and x < W and y >= 0 and y < H

    if len(lines_blens) < 3:
        return False
    elif len(lines_blens) == 3:
        l0 = lines_blens[0][0]
        l1 = lines_blens[1][0]
        l2 = lines_blens[2][0]
        i0 = intersection(l0, l1)
        i1 = intersection(l1, l2)
        i2 = intersection(l2, l0)
        if not valid(*i0): return [l2, l0, l1, i2, i1] #[base, side1, side2]
        if not valid(*i1): return [l0, l2, l1, i2, i0]
        if not valid(*i2): return [l1, l2, l0, i1, i0]
        x, y = i0
        ngbh = img[y-5:y+5, x-5:x+5] #TODO: unsafe
        if np.sum(ngbh) == 0:
            return [l2,l0,l1, i2, i2]
        x, y = i1
        ngbh = img[y-5:y+5, x-5:x+5] #TODO: unsafe
        if np.sum(ngbh) == 0:
            return [l0,l2,l1, i2, i0]
        return [l1, l2, l0, i1,i0]
    else:
        base = max(lines_blens, key = lambda line_blen: line_blen[1])[0]
        x1, x2, y1, y2 = base
        new_lines_i = []
        for line, _ in lines_blens:
            if line == base: continue
            x, y = intersection(base, line)
            x, y = int(x), int(y)
            if valid(x,y):
                ngbh = img[y-5:y+5, x-5:x+5] #TODO: unsafe
                if np.sum(ngbh) > 0:
                    x3,x4,y3,y4 = line
                    v1 = np.array([x2-x1, y2-y1])
                    v2 = np.array([x4-x3, y4-y3])
                    cosine = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                    print(v1, v2, line, cosine)
                    if abs(cosine) < 1/2:
                        new_lines_i.append((line, np.array([x,y]), np.sum(ngbh)))
        if len(new_lines_i) < 2:
            return False
            #randoms = [line for line, _ in lines_blens if line != base]
            #return [base, randoms[0], randoms[1]] 
        new_lines_i = sorted(new_lines_i, key = lambda lps: lps[2])[::-1]
        side1, i1 = new_lines_i[0][0], new_lines_i[0][1]
        if np.linalg.norm(new_lines_i[1][1] - new_lines_i[0][1]) > 50 or len(new_lines_i) < 3:
            side2, i2 = new_lines_i[1][0], new_lines_i[1][1]
        else:
            side2, i2 = new_lines_i[2][0], new_lines_i[2][1]
        return [base, side1, side2, i1, i2]
            

def polar2cartesian(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return [x1,x2,y1,y2]

def get_intersection(img, side, base_int):
    x1,x2,y1,y2 = side
    dir = np.array([x2-x1, y2-y1])
    dir = 15 * dir/ np.linalg.norm(dir)
    p = dir + base_int
    x, y = int(p[0]), int(p[1])
    #img = np.array(img * 0.25, dtype=np.uint8)
    #img[y-5:y+5, x-5:x+5] = 255
    #show(img)
    print('sum', np.sum(img[y-5:y+5, x-5:x+5]))
    if np.sum(img[y-5:y+5, x-5:x+5]) == 0:
        dir = -1 * dir
    dir = dir/3
    p = base_int
    while True:
        p = p + dir
        x, y = int(p[0]), int(p[1])
        s = np.sum(img[y-5:y+5, x-5:x+5])
        if s == 0:
            return (x,y)

def get_border_points(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = np.array(cv2.HoughLines(edges,1,np.pi/180,50))
    lines = lines[:, 0]
    lines = [polar2cartesian(rho,theta) for rho,theta in lines]
    lines = cluster_lines(lines)
    lines_blens = []
    for line in lines:
        good, border_len = good_line(img, *line)
        if good:
            lines_blens.append((line,border_len))
            print(line,border_len)
    tl = three_line(lines_blens, img)
    if tl:
        base, side1, side2 = tl[:3]
        i1, i2 = tl[3:]
        i3 = get_intersection(img, side1, i1)
        i4 = get_intersection(img, side2, i2)
        print(i3, i4, i1, i2)
        return [i3, i4, i1, i2]
    return False


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
    print("\nProcessing ", name)


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
    M = cv2.getRotationMatrix2D(tuple(center),angle,1)
    size = max(img.shape)
    img = cv2.warpAffine(img, M, (size,size))

    # Szukanie podstawy i boków
    points = get_border_points(img)
    H,W = img.shape
    nz = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(nz)
    img = rotate_to_common(img, h, w)
    show(img)
    if points:
        nz = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(nz)
        points = np.float32(points)
        pts2 = np.float32([[W//2, H//2], [W//2+w, H//2], [W//2, H//2+h], [W//2+w, H//2+h]])
        print(points)
        print(pts2)
        matrix = cv2.getPerspectiveTransform(points, pts2)
        img = cv2.warpPerspective(img, matrix, (W+W//2, H+H//2))
        show(img)

    nz = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(nz)
    img = cv2.Canny(img,50,150,apertureSize = 3)
    img = img[y:y+h, x:x+w]



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
    return contour.argmax(0)


def binarize(img):
    img[img!=0] = 255
    return img


def compare(chars, rev_chars):
    similarity_mat = np.zeros((len(chars), len(chars)))
    norms = [np.linalg.norm(vec) for vec in chars]
    rev_norms = [np.linalg.norm(vec) for vec in rev_chars]
    for i in range(len(chars)):
        for j in range(len(chars)):
            if i == j: continue
            if norms[i] < 1 or norms[j] < 1: continue
            score1 = chars[i][::-1].dot(chars[j])/(norms[i]*norms[j])
            score2 = rev_chars[i].dot(chars[j])/(rev_norms[i]*norms[j])
            similarity_mat[i][j] = max(score1, score2)
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
    path = f'daneB/set{sys.argv[1]}/'
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
