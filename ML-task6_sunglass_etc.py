#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2


# In[5]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[6]:


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[7]:


cap = cv2.VideoCapture(0)


# In[8]:


sunglasses = cv2.imread('sunglasses.png', -1)


# In[9]:


logo_top_left = cv2.imread('logo_top_left.png')
logo_top_left = cv2.resize(logo_top_left, (150, 150))  # Resize the logo image
logo_top_right = cv2.imread('logo_top_right.png')
logo_top_right = cv2.resize(logo_top_right, (150, 150))  # Resize the logo image


# In[ ]:


while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the region of interest (ROI) for eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Perform eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Iterate over the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Calculate the coordinates for placing the sunglasses on the eyes
            sx = x + ex
            sy = y + ey
            sw = ew
            sh = eh

            # Resize the sunglasses image to fit the size of the eyes
            sunglasses_resized = cv2.resize(sunglasses, (sw, sh), interpolation=cv2.INTER_AREA)

            # Overlay the sunglasses on the eyes
            for i in range(sh):
                for j in range(sw):
                    if sunglasses_resized[i, j][3] != 0:  # Check the alpha channel
                        frame[sy + i, sx + j] = sunglasses_resized[i, j, :-1]

    # Add the logo images in the corners
    logo_height, logo_width, _ = logo_top_left.shape
    frame[10:10 + logo_height, 10:10 + logo_width] = logo_top_left
    frame[10:10 + logo_height, frame.shape[1] - 10 - logo_width:frame.shape[1] - 10] = logo_top_right

    # Add fixed text position
    cv2.putText(frame, "Vimal Daga Sir's Student and ARTH 3.0 Learner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Cool Accessories, Logo, and Text Face Detection', frame)

    # Exit the loop if
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:


cap.release()


# In[ ]:


cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




