import cv2
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression



DEBUG = False
DEBUGH = True
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'
eastPath = "frozen_east_text_detection.pb\\frozen_east_text_detection.pb"

# set basic params --> using pyimagetutorial
minConfidence = 0.5

# EAST requires multiples of 32
width = 320
height = 320
# newH = 0
# newW = 0
# orig = None


def east_preprocess(img):
    image = cv2.imread(img)
    orig = image.copy()
    # here we grab image dimensions
    (H, W) = image.shape[:2]

    # set w and h and apply ratio
    rW = W / float(width)
    rH = H / float(height)

    # now we resize image
    image = cv2.resize(image, (int(rW), int(rH)))
    (H, W) = image.shape[:2]

    return [image, [H, W], [rW, rH], orig]


def extract_feature_map(img, fh, fw):
    # first name gives us probability of text in region and second is geometry to derive bounding boxes
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet(eastPath)

    # construct blob
    blob = cv2.dnn.blobFromImage(img, 1.0, (fw, fh), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < minConfidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append(startX, startY, endX, endY)
            confidences.append(scoresData[x])

    return [rects, confidences]


def result(rects, conf, newW, newH, orig):
    boxes = non_max_suppression(np.array(rects), probs=conf)

    for (sX, sY, eX, eY) in boxes:
        sX = int(sX * newW)
        sY = int(sY * newH)
        eX = int(eX * newW)
        eY = int(eY * newH)

        cv2.rectangle(orig, (sX, sY), (eX, eY), (0, 255, 0), 2)

    cv2.imshow("text-detect", orig)
    cv2.waitKey(0)


inputImg = 'DJI_0001.JPG'

eastRet = east_preprocess(inputImg)

extractRet = extract_feature_map(eastRet[0], eastRet[1][0], eastRet[1][1])

result(extractRet[0], extractRet[1], eastRet[2][0], eastRet[2][1], eastRet[3])



