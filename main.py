import pytesseract
from pytesseract import Output
import PIL.Image
import cv2

"""
 install tesseract
 tesseract-ocr:add tesseract folder to path and tessdata folder to TESSDATA_PREFIX
 pip install pytesseract
 cmd --> tesseract ImageName.jpg stdout

 to use mode:
 cmd --> tesseract ImageName.jpg stdout --psm number_of_mode

 to save text:
 cmd --> tesseract ImageName.jpg text.txt


"""

"""
 Page segmentation modes:
  0     Orientation and script detection (OSD) only.
  1     Automatic page segmentation with OSD.
  2     Automatic page segmentation, but no OSD, or OCR.
  3     Fully automatic page segmentation,but no OSD. (Default)
  4     Assume a single column of text of variable sizes.
  5     Assume a single uniform block of vertically aligned text.
  6     Assume a single uniform block of text.
  7     Treat the image as a single text line.
  8     Treat the image as a single word.
  9     Treat the image as a single word in a circle.
  10    Treat the image as a single character.
  11    Sparse text. Find as much as possible in no particular order.
  12    Sparse text with OSD.
  13    Raw line. Treat the image as single text line, 
                        bypassing hacks that are Tesseract-specific.
"""

"""
OCR Engine Mode
 0      Legacy engine only.
 1      Neural nets LSTM engine only.
 2      Legacy + LSTM engines.
 3      Default, based on what is available.
"""

myconfig = r"--psm 11 --oem 3"

img = cv2.imread("rumi1.jpg")
height, width, _ = img.shape

data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)

amount_boxes = len(data['text'])
for i in range(amount_boxes):
    if float(data['conf'][i]) > 80:
        (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
        img = cv2.putText(img, data['text'][i], (x, y+height+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0), 1, cv2.LINE_AA)



cv2.imshow("img", img)
cv2.waitKey(0)
