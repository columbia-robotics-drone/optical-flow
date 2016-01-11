__author__ = 'Robert Kwiatkowski'
import numpy
import cv2
import srinivasan as sr


def capture():
    cap = cv2.VideoCapture(-1)
    last_frame = None
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        this_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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