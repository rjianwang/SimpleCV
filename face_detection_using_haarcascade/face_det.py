#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

def face_det(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)

    # detect faces
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    # draw rectangles around faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # show results
    cv2.imshow('Face Detection', img)
    while cv2.waitKey(0) == 'q':
        break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('lena.jpg')
    #img = cv2.imread('liushishi.jpg')
    #img = cv2.imread('liushishi2.jpg')
    #img = cv2.imread('liushishi3.jpg')
    face_det(img)

