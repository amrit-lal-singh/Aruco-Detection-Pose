import numpy as np
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

# Here we store the distortion coefficients
dist = np.array(([[-2.65932816e-01, 3.04981998e+00, -1.31423517e-03, -6.12105921e-04,
                   -1.19332532e+01]]))

# here we store the camera matrix
mtx = np.array([[976.9729196, 0., 524.00069881],
                [0., 974.93729777, 362.12927844],
                [0., 0., 1.]])

while True:
    flag, frame = cap.read()

    # size pictures
    # frame=cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
    # Grayscale words
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Initialize detector parameters with default values
    parameters = aruco.DetectorParameters_create()
    #Returns the ID and the coordinates of the four corners of the sign board
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                                                          parameters=parameters, cameraMatrix=mtx,
                                                          distCoeff=dist)
    # Draw the position of the sign
    aruco.drawDetectedMarkers(frame, corners, ids)

    marker_length = 10.00  # mm

    corners = np.array(corners)
    print(corners)


    print("end")
    if corners != []:
        pose = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        # frame = draw_axis(frame,rvec, tvec, camMat, distCoeffs)
        print(rvec)
        print("start")
        #(rvec - tvec).any()  # get rid of that nasty numpy value array error
        cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, length=0.05)
        print(tvec)  # get tvec

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()
