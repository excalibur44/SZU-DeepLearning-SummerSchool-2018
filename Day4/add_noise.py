# coding: utf-8
import os
import cv2
import random
from numpy import shape
from numpy.random import random_integers

def PepperandSalt(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random. randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        if random_integers(0, 1) <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg

def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        NoiseImg[randX, randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)
        if  NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg

def add_noise(filename, path=''):
    img = cv2.imread(path + filename, 0)
    img_ps = PepperandSalt(img, 0.02)
    img_gs = GaussianNoise(img, 2, 4, 0.02)
    cv2.imwrite(path + "ps_" + filename, img_ps)
    cv2.imwrite(path + "gs_" + filename, img_gs)
    #cv2.imshow('PepperandSalt', img_ps)
    #cv2.imshow('Gaussian', img_gs)
    #cv2.waitKey(0)

path = 'training'
with open("BuildLMDB/" + path + ".txt", 'a+') as f:
    for d in os.listdir(path):
        for filename in os.listdir(path + '/' + d):
            add_noise(filename, path + '/' + d + '/')

path = 'testing'
with open("BuildLMDB/" + path + ".txt", 'a+') as f:
    for d in os.listdir(path):
        for filename in os.listdir(path + '/' + d):
            add_noise(filename, path + '/' + d + '/')
