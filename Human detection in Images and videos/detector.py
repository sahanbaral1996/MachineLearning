from skimage.feature import hog
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import cv2
np.set_printoptions(suppress=True)


def slide(img):
    for y in range(0,img.shape[0],10):
        for x in range(0,img.shape[1],10):
            yield (x, y, img[y: y + 128, x: x + 64])

def detect(img):
    im = imutils.resize(img, width=min(400, img.shape[1]))
    detections= []
    scale = 0
    model = joblib.load("model/svm_data.model")
    for im_scaled in pyramid_gaussian(img, downscale=1.6):
        if im_scaled.shape[0] < 128 or im_scaled.shape[1] < 64:
            break
        for (x, y, im_window) in slide(img):
            if im_window.shape[0] != 128 or im_window.shape[1] != 64:
                continue
            hist = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L1',
                   transform_sqrt=False, feature_vector=True, visualize=False)
            hist = hist.reshape(1,-1)
            if model.predict(hist) == 1 :
                if model.decision_function(hist) > 0.28:
                    detections.append((int(x * (1.6 ** scale)), int(y * (1.6 ** scale)),
                                       model.decision_function(hist), int(64 * (1.6 ** scale)),
                                       int(128* (1.6 ** scale))))
        scale += 1
    clone = img.copy()

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    # print ("shape, ", pick.shape)

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()



if __name__== "__main__":
    img = cv2.imread("4.jpg", cv2.IMREAD_GRAYSCALE)
    detect(img)