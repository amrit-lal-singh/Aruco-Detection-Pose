import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2.aruco as aruco



def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M
im = cv2.imread("persp.jpeg")
w, h = im.shape[0], im.shape[1]
# We will first manually select the source points
# we will select the destination point which will map the source points in
# original image to destination points in unwarped image




frame = im
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(grey, (5, 5), 0)
grey = cv2.fastNlMeansDenoising(grey)
mask = cv2.inRange(grey,110,255)
edges = cv2.Canny(mask, 190, 200)
corners = cv2.goodFeaturesToTrack(edges, 4, 0.5, 300)
print(corners)
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180,  threshold=100,  minLineLength=100, maxLineGap=10)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key = cv2.contourArea)

contourImg = cv2.drawContours(frame, contour, -1, (0,255,0), 3)
cv2.imshow("Contours", contourImg)

for points in lines:

    x1, y1, x2, y2 = points[0]

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


f = [0,0]
e = [3000, 3000]


for x in range(w):
    for y in range(h):
        if edges[x][y] > 125:
            if x+y > f[0]+f[1]:
                f[0] = x
                f[1] = y
        if edges[x][y] > 125:
            if x+y < e[0]+e[1]:
                e[0] = x
                e[1] = y


print(edges)
print(f,e)

cv2.imshow('mask', mask)
cv2.imshow('edges', frame)
cv2.imshow('kidio', edges)


src = np.float32([(967, 550),

 ( 775, 183),

 (652, 530),

 (488, 274)])


dst = np.float32([(600, 0),
                  (0, 0),
                  (600, 531),
                  (0, 531)])

unwarp(im, src, dst, True)


cv2.imshow("so", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
