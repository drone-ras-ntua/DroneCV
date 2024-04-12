import cv2 as cv
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import numpy as np
import os

def qr_recognize(image_path):

    drone_landing = False

    if  not os.path.exists(image_path):
        print("Image path does not exist")
        return None, False, None

    image = cv.imread(image_path) # Read the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert the image to grayscale
    decoded_objects = decode(gray) # Decode QR codes

    for obj in decoded_objects:

        data = obj.data.decode('utf-8') # Extract data and bounding box coordinates of the QR code

        if data == 'RAS-ICARUS FLYING TEAM':

            drone_landing = True
            x, y, w, h = obj.rect
            qr_center = np.array([ (x + (w/2)), (y + (h/2)) ])

            return decoded_objects, drone_landing, qr_center
    
        else:
            return None, False, None

if __name__ == '__main__':

    img_file_name = "landing.jpg"
    image_path = os.path.join('Landing_QR_Script/img', img_file_name)

    qr_infos, landing_status, qr_center = qr_recognize(image_path)

    #CHECK
    if landing_status:
        image = cv.imread(image_path)
        plt.figure(figsize= [10, 10])
        plt.imshow(image)
        plt.scatter(qr_center[0], qr_center[1], color = 'Red')
        plt.title('LANDING VIEW')
        plt.show()


