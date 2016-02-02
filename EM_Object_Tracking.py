__author__ = 'robertk'
import os, sys
from PIL import Image
import numpy as np
import cv2
import math
import time
import datetime
import random
import cv

def get_pixel_values(img_path):
    im = cv2.imread(img_path, 0)
    return im

def find_borders(pix, sensitivity):
    l, w = pix.shape
    m = 0
    var = 0
    for y in range(l-1):
        for x in range(w-1):
            m += abs(int(pix[y][x])-int(pix[y][x+1]))
            m += abs(int(pix[y][x])-int(pix[y+1][x]))

    m = m/(2*(l-1)*(w-1))
    for y in range(l-1):
        for x in range(w-1):
            var += (m-abs(int(pix[y][x])-int(pix[y][x+1])))**2
            var += (m-abs(int(pix[y][x])-int(pix[y+1][x])))**2
    var = var/(2*(l-1)*(w-1))
    sd = math.sqrt(var)

    list = []

    n = 2/sensitivity

    for y in range(l-1):
        for x in range(w-1):
            if abs(int(pix[y][x])-int(pix[y][x+1])) > m+n*sd or abs(int(pix[y][x])-int(pix[y+1][x])) > m+n*sd:
                pix[y][x] = 0
                list.append((float(y)/float(l), float(x)/float(w)))
            else:
                pix[y][x] = 255
    data = np.asarray(list)
    return pix, data

def find_centers_from_pixels(means, pix, max_iter):
    m, d = means.shape
    l, w = pix.shape
    i = 0
    data = np.zeros((pix.size, d))
    for y in range(l):
        for x in range(w):
            data[i] = (x/w, y/l, pix[y][x]/256)
            i += 1
    return find_centers(means, data, max_iter)

def initEM(m, data): # assumes data ranges from 0-1
    n, d = data.shape
    cov = np.zeros((d*m, d))
    for i in range(m):
        cov[i*d:(i+1)*d, 0:d] = np.identity(d)*((1/float(m))**2)
    mix = np.ones((m, 1))*(1/float(m))
    means = np.random.rand(m, d)
    return means, cov, mix

def find_centers(m, d, data, max_iter):
    path = []
    converged = False
    thresh = 0.00001
    #m, d = means.shape
    #mix = np.ones((m, 1))*(1/m)
    #cov = np.zeros((d*m, d))
    means, cov, mix = initEM(m, data)
    #for i in range(m):
    #    cov[i*d:(i+1)*d, 0:d] = np.identity(d)*random.random()
    i = 0
    ll = -1000000
    while i < max_iter and not converged:
        path.append(means)
        prev = ll
        tau, ll = E_step(means, cov, mix, data)
        print('Loss: {0}'.format(ll))
        if ll-prev <= thresh:
            converged = True
        means, cov, mix = M_step(tau, data)
        i += 1
    print("Done")
    return means, cov, path

def E_step(means, cov, mix, data):
    print('E {0}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')))
    m, d = means.shape
    tau = np.zeros((len(data), m))
    ll = 0
    for i in range(m):
        cov_m = cov[i*d:(i+1)*d, 0:d]
        detcov = np.linalg.det(cov_m)
        invcov = np.linalg.inv(cov_m+np.identity(d)*0.00001)
        coef = mix[i]*((2*math.pi)**(-d/2))*(detcov**(-0.5))

        for n in range(len(data)):
            diff = np.matrix(data[n]-means[i])
            tau[n][i] = coef*math.exp(-0.5*(diff*invcov*np.transpose(diff)))

    sumtau = np.zeros(m)
    for n in range(len(data)):
        l = 0
        for i in range(m):
            l += tau[n][i]
        for i in range(m):
            tau[n][i] /= l
            #sumtau[i] += tau[n][i]

        ll += math.log(l)

    return tau, ll

def M_step(tau, data):
    print('M {0}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')))

    n, m = tau.shape
    n, d = data.shape
    means = np.zeros((m, d))
    mix = np.zeros((m, 1))
    cov = np.zeros((d*m, d))
    sumtau = np.zeros((m, 1))
    for i in range(m):
        for n in range(len(data)):
            sumtau[i] += tau[n][i]

    for i in range(m):
        for n in range(len(data)):
            means[i] += tau[n][i]*data[n]
        means[i] /= sumtau[i]
        for n in range(len(data)):
            cov[i*d:(i+1)*d, 0:d] += tau[n][i]*np.transpose(np.matrix(data[n]-means[i]))*np.matrix(data[n]-means[i])
        cov[i*d:(i+1)*d, 0:d] = cov[i*d:(i+1)*d, 0:d]/sumtau[i] + np.identity(d)*0.00001

        mix[i] = sumtau[i]/n
    return means, cov, mix

def test_img(m, img_path):
    print('Start {0}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')))
    d = 2
    means = np.random.rand(m, d)
    #img_path = 'swarm-robotics.jpg'
    pix = get_pixel_values(img_path)

    cv2.imshow('Raw',pix)

    pix, data = find_borders(pix, 0.5)
    l, w = pix.shape

    means, cov, path = find_centers(m, d, data, 100)

    colored = cv2.cvtColor(pix, cv.CV_GRAY2RGB)
    for i in range(m):
        x = means[i][0]*w
        y = means[i][1]*l
        for j in range(-10, 10):
            if (x+j > 0 and x+j < w):
                colored[y][x+j] = (0, 0, 255)
        for j in range(-10, 10):
            if (y+j > 0 and y+j < l):
                colored[y+j][x] = (0, 0, 255)


    cv2.imshow('Borders',pix)
    cv2.imshow('Color And Means', colored)
    print('End {0}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test(m, path):
    fr = open(path)
    str = fr.readline()
    list = []
    l, w = 500, 500
    maxX = 0
    maxY = 0
    while str:
        nums = str.split()
        x = abs(float(nums[0]))
        y = abs(float(nums[1]))
        if x > maxX:
            maxX = x
        if y > maxY:
            maxY = y
        list.append((x, y))
        str = fr.readline()

    maxX *= 1.2
    maxY *= 1.2

    data = np.zeros((len(list), 2))
    colored = np.zeros((l, w, 3))
    for i in range(len(list)):
        x = (.1*maxX+list[i][0])/maxX
        y = (.1*maxY+list[i][1])/maxY
        data[i] = (x, y)
        colored[int(y*l)][int(x*w)] = 255, 255, 255

    d = 2

    means, cov, path = find_centers(m, d, data, 100000)
    for n in range(len(path)):
        for i in range(m):
            colored[path[n][i][0]*w][path[n][i][1]*l] = (0, 255, 0)

    for i in range(m):
        x = means[i][0]*w
        y = means[i][1]*l
        #cov_m = cov[i*d:(i+1)*d, 0:d] = np.identity(d)*((1/float(m))**2)
        #cv2.ellipse(colored, (x, y), (cov_m[0][0], cov_m[1][1]), 0, 360)
        for j in range(-10, 10):
            if (x+j > 0 and x+j < w):
                colored[y][x+j] = (0, 0, 255)
        for j in range(-10, 10):
            if (y+j > 0 and y+j < l):
                colored[y+j][x] = (0, 0, 255)


    cv2.imshow('Color',colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#test(3, 'testB.txt')
test_img(10, 'low_res_roomba.jpg')