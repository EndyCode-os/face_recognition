import face_recognition #use for face detection and recognition
import numpy as np #For handling image data as arrays
import cv2 #for capturing video (like from a webcam) and drawing boxes around faces.
from PIL import Image   # For loading and displaying images 

def recognize_faces(group_image, known_image):
            
    #Convert the images from RBG to BGR 
    face_bgr = cv2.cvtColor(known_image, cv2.COLOR_RGB2BGR)
    group_bgr = cv2.cvtColor(group_image, cv2.COLOR_RGB2BGR)

    # detect the face in the image (give me the coordinates)
    face_coordinates = face_recognition.face_locations(known_image)
    group_face_coordinates = face_recognition.face_locations(group_image)

    #Encoding face
    known_encoding = face_recognition.face_encodings(known_image)
    group_face_encoding = face_recognition.face_encodings(group_image, group_face_coordinates)


    # Compare faces and check for match

    for face_encoding, face_location in zip (group_face_encoding, group_face_coordinates):
        
        face_comparaison = face_recognition.compare_faces(known_encoding, face_encoding)
        distance = face_recognition.face_distance(known_encoding, face_encoding)

        top, right, bottom, left = face_location

        if face_comparaison[0] and distance[0] < 0.6:
            label = "Kamala Harris"
            color = (0, 255, 0)
        
        else:
            label= 'Unknown'
            color = (0, 0, 255)

        cv2.rectangle(group_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(group_bgr, label, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        
    return group_bgr

# read the file, converts it into pixels values, and load into memory
face = face_recognition.load_image_file('face.jpg')
group = face_recognition.load_image_file('group.jpg')

face_matching = recognize_faces(group, face)

#display the image with the rectangle
cv2.imshow("Group Face Detection",face_matching )

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

