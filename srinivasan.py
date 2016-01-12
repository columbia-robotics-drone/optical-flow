__author__ = 'Robert Kwiatkowski'
import numpy

def srinivasan(this_frame, last_frame):
    w, l = this_frame.shape
    A = numpy.zeros((2, 1))
    B = numpy.zeros((2, 2))
    for x in range(1, w-1):
        for y in range(1, l-1):
            # The difference between the values of the pixels to the left and to the right
            p_x = (this_frame[x-1][y]-this_frame[x+1][y])
            # The difference between the values of the pixels to above and below
            p_y = (this_frame[x][y-1]-this_frame[x][y+1])
            # The difference between the values of the pixels and the pixel of the last frame
            p_t = (this_frame[x][y]-last_frame[x][y])

            # First vector, frame difference normalized by x and y respectively
            A[0] += p_t*p_x
            A[1] += p_t*p_y

            # First Matrix, resembles a covariance matrix,
            # | difference of x's squared, difference of x * difference of y |
            # | difference of x * difference of y, difference of y's squared |
            B[0][0] += p_x**2
            B[0][1] += p_x*p_y
            B[1][0] += B[0][1]
            B[1][1] += p_y**2

    # Formatting these matricies so they can be multiplied together
    A = numpy.transpose(2*A)
    B = numpy.linalg.inv(B)

    # Returning the product
    return numpy.dot(A, B)
