import numpy as np
import cv2
import glob

# FOR CAMERA CALLIBRATION

# DEFINE THE COORDINATES OF THE CORNER POINTS OF THE CHESSBOARD IN THE CHESSBOARD REFERENCE FRAME
objp = np.zeros((9 * 7, 3), np.float32)

objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# ACCESSING THE IMAGES TO BE USED FOR CALLIBRATION
images = glob.glob('pics/*.jpg')

# CONDITION TO STOP CHECKING AFTER AN ACCURACY IS REACHED IN CORNER IDENTIFICATION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:

    # READ IMAGE AND FIND THE CORNERS OF THE CHESSBOARD USING INBUILT FUNCTIONS
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, (7, 9),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        # FOR EACH IMAGE, REFINE THE CORNERS AND THEN APPEND THE INTO IMAGE POINTS LIST
        objpoints.append(objp)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # DRAW THE CHESSBOARD CORNERS USING INBUILT FUNCTION
        '''
        img = cv2.drawChessboardCorners(cv2.imread(fname), (9,7), corners,ret)
        cv2.imshow("CHESSBOARD", img)
        cv2.waitKey(500)
        '''

# USING THE INBUILT FUNCTION GENERATE THE CAMERA MATRIX AND THE DISTORTION MATRIX
ret, Camera_matrix, Distortion_matrix, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None,
                                                                          None)

print("Camera Matrix:\n")
print(Camera_matrix)
print("\nDistortion Matrix:\n")
print(Distortion_matrix)
