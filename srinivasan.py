__author__ = 'robertk'
__author__ = 'robertk'
import numpy

def srinivasan(pixel_intensities, last_pixel_intensities):
    w, l = pixel_intensities.shape()
    A = numpy.zeros(2)
    B = numpy.zeros(2, 2)
    for x in range(1, w-1):
        for y in range(1, l-1):
            A[0] += (pixel_intensities(x, y)-last_pixel_intensities(x, y))*(pixel_intensities(x-1, y)-last_pixel_intensities(x+1, y))
            A[1] += (pixel_intensities(x, y)-last_pixel_intensities(x, y))*(pixel_intensities(x, y-1)-last_pixel_intensities(x, y+1))

            B[0][0] += (pixel_intensities(x-1, y)-last_pixel_intensities(x+1, y))**2
            B[0][1] += (pixel_intensities(x-1, y)-last_pixel_intensities(x+1, y))*(pixel_intensities(x, y-1)-last_pixel_intensities(x, y+1))
            B[1][0] += (pixel_intensities(x-1, y)-last_pixel_intensities(x+1, y))*(pixel_intensities(x, y-1)-last_pixel_intensities(x, y+1))
            B[1][1] += (pixel_intensities(x, y-1)-last_pixel_intensities(x, y+1))**2
    return numpy.multiply(2*A, B)