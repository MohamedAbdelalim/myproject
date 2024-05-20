import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

class PlateRecognitionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.data:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        

        # Read Image, grayscale and blur
        img = cv2.imread('image')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying filter and edges for localization 
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        # Contour dedaction and mask applying and returning top 10 contours
        Keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(Keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Lopping through top10 contoursand checking for 4 key points in them
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        # Apply masking and isolate location
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Isolate only plate to pass it to easy OCR 
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Read plate
        reader = easyocr.Reader(['en','ar'])
        result = reader.readtext(cropped_image)

        # Render result
        text = result[0][-2]
        print(text)

        return Response({"text": text}, status=status.HTTP_200_OK)
