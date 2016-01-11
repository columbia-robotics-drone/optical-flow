__author__ = 'Robert Kwiatkowski'
import numpy

def srinivasan(this_frame, last_frame):
    w, l = this_frame.shape
    A = numpy.zeros(2)
    B = numpy.zeros((2, 2))
    for x in range(1, w-1):
        for y in range(1, l-1):
            A[0] += (this_frame[x][y]-last_frame[x][y])*(this_frame[x-1][y]-this_frame[x+1][y])
            A[1] += (this_frame[x][y]-last_frame[x][y])*(this_frame[x][y-1]-this_frame[x][y+1])

            B[0][0] += (this_frame[x-1][y]-last_frame[x+1][y])**2
            B[0][1] += (this_frame[x-1][y]-last_frame[x+1][y])*(this_frame[x][y-1]-last_frame[x][y+1])
            B[1][0] += (this_frame[x-1][y]-last_frame[x+1][y])*(this_frame[x][y-1]-last_frame[x][y+1])
            B[1][1] += (this_frame[x][y-1]-last_frame[x][y+1])**2
    return numpy.multiply(2*A, B)
