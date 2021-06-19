#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install pytesseract')
get_ipython().system('{sys.executable} -m pip install matplotlib')


# In[4]:


import pytesseract
import cv2
import matplotlib.pyplot as plt


# # For configuration

# In[5]:


pytesseract.pytesseract.tesseract_cmd=r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# In[6]:


img=cv2.imread(r'C:\Users\HP\Desktop\OCR\Demo.png')


# In[ ]:


plt.imshow(img)


# In[8]:


img2char= pytesseract.image_to_string(img)


# In[ ]:


print("Text form of data :","\n",img2char)


# In[10]:


imgbox=pytesseract.image_to_boxes(img)


# In[ ]:


print(imgbox)


# In[12]:


imgh,imgw,_=img.shape


# In[13]:


img.shape


# In[14]:


for boxes in imgbox.splitlines():
    boxes=boxes.split(" ")
    x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
    cv2.rectangle(img, (x,imgh-y),(w, imgh-h) , (0,0,255),3)


# In[ ]:


plt.imshow(img)


# In[ ]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# # Video Demo

# In[17]:



font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

cap=cv2.VideoCapture(r"C:\Users\HP\Desktop\OCR\Final.mp4")
#cap.set(cv2.CAP_PROP_FPS,170)
#Check if the video is opened correctly , else display a message
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
cntr=0;
while True:
    ret,frame=cap.read()
    cntr=cntr+1
    if((cntr%20)==0):
        imgh,imgw,_=frame.shape
        x1,y1,w1,h1=0,0,imgh,imgw
        imgchar=pytesseract.image_to_string(frame)
        imgboxes=pytesseract.image_to_boxes(frame)
        for boxes in imgboxes.splitlines():
            boxes=boxes.split(" ")
            x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
            cv2.rectangle(frame,(x,imgh-y),(w,imgh-h),(0,0,255),3)

        cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
       
        
        cv2.imshow('Text Detection Tutorial', frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
            


# # Web Camera

# In[26]:



font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

cap=cv2.VideoCapture(1)
#cap=cv2.VideoCapture(r"C:\Users\HP\Desktop\OCR\Final.mp4")
#cap.set(cv2.CAP_PROP_FPS,170)#
#Check if the webcam is opened correctly , else display a message
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Camera")
cntr=0;
while True:
    ret,frame=cap.read()
    cntr=cntr+1
    if((cntr%20)==0):
        imgh,imgw,_=frame.shape
        #eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades_eye.xml')
        x1,y1,w1,h1=0,0,imgh,imgw
        imgchar=pytesseract.image_to_string(frame)
        imgboxes=pytesseract.image_to_boxes(frame)
        for boxes in imgboxes.splitlines():
            boxes=boxes.split(" ")
            x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
            cv2.rectangle(frame,(x,imgh-y),(w,imgh-h),(0,0,255),3)
        #cv2.rectangle(frame , (x1,x1),(x1+w1, y1+h1),(0,0,0),-1)
            #Add text
        cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        #font= cv2.FONT_HERSHEY.SIMPLEX
           
            
        #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(faceCascade.empty())
        #faces=faceCascade.detectMultiScale(gray,1.1,4)
        
        #Draw a rectangle around the face
        #for(x,y,w,h) in  faces:
            #cv2.recatngle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
        #Use putText() method for
        #inserting text on video
        
        cv2.imshow('Text Detection Tutorial', frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
            


# In[ ]:





# In[ ]:





# In[ ]:




