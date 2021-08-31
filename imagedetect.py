import cv2
import pytesseract
import numpy as np



DEBUG = False
DEBUGH = True
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

# sort contour NEEDS ADJUSTMENT
def contour_key(contour):

    #min bb area for contour
    rect = cv2.minAreaRect(contour)

    w, h = rect[1]
    if w < h:
        w, h = h, w

    return w * h


def get_contours(img):
    # Remove noise with median blurring
    image = cv2.medianBlur(img, 5)

    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run canny edge detector on the grayscale image
    edge = cv2.Canny(grayscale, 100, 200)

    # Dilate the edge image to get a blob of text
    kernel = np.ones((1, 5), dtype=np.uint8)
    dilated = cv2.dilate(edge, kernel, iterations=10)

    if DEBUG:
        cv2.imwrite("/tmp/edge.jpg", edge)
        cv2.imwrite("/tmp/dilated.jpg", dilated)

    # Find the top 10 contours based on the area of their bounding rectangles
    contours = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=contour_key, reverse=True)[:10]

    return contours


def get_bounding_boxes(contours, image):
    boxes = []

    if DEBUG:
        image_copy = image.copy()

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        boxes.append(rect)

        if DEBUG:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image_copy, [box], 0, (0, 255, 0), 3)

    if DEBUG:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 700, 700)
        cv2.imshow('result', image_copy)
        cv2.waitKey(0)

    return boxes


def ocr_int(i, box, image):
    center = box[0] # Center of the bounding rectangle
    w, h   = box[1] # Width and height of the bounding rectangle
    angle  = box[2] # Angle of the bounding rectangle
    if w < h:
        w, h = h, w
        angle += 90.0

    rows, cols, _ = image.shape

    # Rotate image
    M       = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Crop rotated image.
    # Ensure that the crop region lies within the image.
    start_x = int(center[1] - (h * 0.6 / 2))
    end_x   = int(start_x + h * 0.6)
    start_y = int(center[0] - (w * 0.6 / 2))
    end_y   = int(start_y + w * 0.6)
    start_x = start_x if 0 <= start_x < rows else (0 if start_x < 0 else rows-1)
    end_x   = end_x if 0 <= end_x < rows else (0 if end_x < 0 else rows-1)
    start_y = start_y if 0 <= start_y < cols else (0 if start_y < 0 else cols-1)
    end_y   = end_y if 0 <= end_y < cols else (0 if end_y < 0 else cols-1)
    crop    = rotated[start_x:end_x, start_y:end_y]

    # Convert to grayscale
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = 0

    # Convert to a binary image using adaptive thresholding
    ret, threshold = cv2.threshold(grayscale, 210, 255, cv2.THRESH_BINARY)

    if DEBUGH:
        cv2.imwrite(r'C:\Users\patri\PycharmProjects\ImageDetection\tmp\thresh_' + str(i) + ".bmp", threshold)

    text = pytesseract.image_to_string(threshold, lang='eng', config='--psm 6')
    print(text)

    return text


# Perform OCR on all the bounding boxes
def ocr(boxes, image):
    for i, box in enumerate(boxes):
        print(i)
        try:
            text = ocr_int(i, box, image)
        except KeyboardInterrupt:
            raise
        except:
            text = None

    return None

# run the scripts
image = cv2.imread('DJI_0001.JPG')

contours = get_contours(image)

boxes = get_bounding_boxes(contours, image)

text = ocr(boxes, image)
