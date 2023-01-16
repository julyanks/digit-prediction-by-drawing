import pygame,sys
from pygame.locals import *
import numpy as np
import keras
from keras.models import load_model
import cv2
from gtts import gTTS  
from playsound import playsound
import os 

windowsx=640
windowsy=480

BOUNDARY =5
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)

MODEL=load_model("C://programming_course//.vscode//semproject//final.h5")

LABELS={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",
        6:"Six",7:"Seven",8:"Eight",9:"Nine"}

pygame.init()
FONT=pygame.font.Font("freesansbold.ttf",18)
displaysurf=pygame.display.set_mode((windowsx,windowsy))

pygame.display.set_caption("Digit Prediction by Drawing")

iswriting=False
number_xcord=[]
number_ycord=[]
imgcnt=1
IMAGESAVE=False
PREDICT=True


while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(displaysurf,WHITE,(xcord,ycord),4,0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type==MOUSEBUTTONDOWN:
            iswriting=True
        
        if event.type==MOUSEBUTTONUP:
            iswriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)

            rect_min_x,rect_max_x=max(number_xcord[0]-BOUNDARY,0),min(windowsx,number_xcord[-1]+BOUNDARY)
            rect_min_y,rect_max_y=max(number_ycord[0]-BOUNDARY,0),min(number_ycord[-1]+BOUNDARY,windowsy)

            number_xcord=[]
            number_ycord=[]

            img_arr=np.array(pygame.PixelArray(displaysurf))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
        
            if IMAGESAVE:
                cv2.imwrite('image.png')
                imgcnt+=1

            if PREDICT:
                images=cv2.resize(img_arr,(28,28))
                images=np.pad(images,(10,10),'constant',constant_values=0)
                images=cv2.resize(images,(28,28))/255
            
                labels=str(LABELS[np.argmax(MODEL.predict(images.reshape(1,28,28,1)))])

                textsurf=FONT.render(labels,True,RED,WHITE)
                textrecobj=textsurf.get_rect()
                textrecobj.left,textrecobj.bottom=rect_min_x,rect_max_y

                displaysurf.blit(textsurf,textrecobj)
                language='en'
                obj = gTTS(text=labels, lang=language, slow=False)  
                obj.save("audio.mp3")  
                os.system("start audio.mp3")

            if event.type==KEYDOWN:
                if event.unicode=='n':
                    displaysurf.fill(BLACK)

        pygame.display.update()
        



