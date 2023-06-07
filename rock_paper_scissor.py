# import the opencv library
import cv2
import tensorflow as t
import numpy as np

model = t.keras.models.load_model("converted_keras/keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame=cv2.resize(frame,(224,224))
    arr = np.array(frame,dtype=np.float32)
    arr = np.expand_dims(arr,axis=0)
    arr = arr/255.0
    pre = model.predict(arr)
    print(pre)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()