__author__ = 'Robert Kwiatkowski'
import numpy
import cv2
import srinivasan as sr

# returns the central w by l section around the midpoint
def subsection(w, l, picture):
    sub = numpy.zeros((w, l))
    w0, l0 = picture.shape
    for x in range(w):
        for y in range(l):
            sub[x][y] = picture[(w0-w)/2+x][(l0-l)/2+y]
    return sub

def capture():
    cap = cv2.VideoCapture(-1)
    last_frame = None
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        this_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Desired Size of the Image
        x, y = 256, 256

        # Reshaping the Image to the desired size
        w, l = this_frame.shape
        this_frame = this_frame[(w-x)/2:(w+x)/2, (l-y)/2:(l+y)/2]

        #this_frame = subsection(480, 480, this_frame)

        # Display the resulting frame
        cv2.imshow('frame',this_frame)

        if last_frame is not None:
            print(sr.srinivasan(this_frame, last_frame))
        last_frame = this_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

capture()