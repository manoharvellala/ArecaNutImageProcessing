from tkinter import *
import tkinter as tk
import cv2
import os
import math
from tkinter import filedialog
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile                            

import imutils
from PIL import Image
from PIL import ImageTk
from sklearn.model_selection import KFold
# global variables

import time
from PIL import ImageTk, Image

from skimage.filters import median
import pandas as pd
global rep
import csv
import copy
import random
from numpy import load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
from numpy import save
from keras.utils import np_utils
import os
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from main_features import texture
from skimage.color import rgb2gray
import pickle
from tkinter import messagebox
from PIL import ImageTk, Image
    
def load():
    filename = filedialog.askopenfilename(title='open')
    main_img = cv2.imread(filename)
    image= cv2.imread(filename)
    image=cv2.resize(image,(256,256))
    return image

def preprocess(image):
    image=cv2.resize(image,(256,256))
    cv2.imshow('Original Image',image)
    median = cv2.medianBlur(image,5)
    cv2.imshow('Median Filtered Image',median)
    cv2.imwrite("median.jpg", median)
    return median

def segmentation(median):
    img = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(img)
    def filter_image(img,mask):
        r = image[:,:,0]*mask
        g = image[:,:,1]*mask
        b = image[:,:,2]*mask
        return np.dstack([r,g,b])
    img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Otsu Threshold', thresh1)
    cv2.imwrite("otsu.jpg",thresh1)
    return thresh1

def features(thresh1):
    x=texture(thresh1)
    z=x
    print(z)
    return z

def classification(z):
    clas=['diseased','normal']
    file = "finalized_model_DT.sav"
    clf_model = pickle.load(open(file, 'rb'))
    pred=clf_model.predict(z.reshape(1,20))
    print((pred))
    print('Given Areakanut is : '+clas[int(pred)])
    res = clas[int(pred)]
    messagebox.showinfo('given arecanut is: ',res)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg='#CCFF99')

        # changing the title of our master widget      
        self.master.title("ARECANUT DISEASE DETECTION")
        
        self.pack(fill=BOTH, expand=1)
        w = tk.Label(root, 
		 text=" ARECANUT DISEASE DETECTION ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=350, y=0)
        # creating a button instance
        quitButton = Button(self,command=self.query, text="LOAD IMAGE",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=50)
        quitButton = Button(self,command=self.preprocess,text="PREPROCESSING",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=100)
        quitButton = Button(self,command=self.segment, text="SEGMENTATION",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=150)
        quitButton = Button(self,command=self.feature,text="FEATURE EXTRACTION",activebackground="dark red",fg="blue",width=20)
        quitButton.place(x=10, y=200)
        quitButton = Button(self,command=self.classification,text="classification",activebackground="dark red",fg="blue",width=20)
        quitButton.place(x=10, y=250)
        quitButton = Button(self,command=self.destroy,text="QUIT",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=300)
        
##        load = Image.open("logo.jfif")
##        render = ImageTk.PhotoImage(load)
##        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
##        image1.image = render
##        image1.place(x=10, y=90)

        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=250, y=50)

        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=500, y=50)

        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image4.image = render
        image4.place(x=750, y=50)
        
#       2nd row

        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image5.image = render
        image5.place(x=250, y=270)

        #image6=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        #image6.image = render
        #image6.place(x=500, y=270)

        #image7=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        #image7.image = render
        #image7.place(x=750, y=270)

#       3rd column
        #variable = StringVar(self)
        #variable.set("SELECT FEATURE TYPE") # default value
        #wi = OptionMenu(self, variable, "Color Features", "Histogram Feature", "GLCM Features","ALL Features",command=callback)
        #wi.pack()
        #wi.place(x=980, y=50)
        contents ="  Waiting for Results..."
        global T
        T = Text(self, height=19, width=25)
        T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        
#       3rd row
        #image5.place(x=300, y=490)

#       Functions

    def query(self, event=None):
        contents ="Loading Image..."
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        rep = filedialog.askopenfilenames()
        # Image operation using thresholding 
        img = cv2.imread(rep[0])
        #cv2.imshow('fff2',img)
        img = cv2.resize(img,(256,256))
        #cv2.imshow('fff1',img)
        Input_img=img.copy()
        print(rep[0])
        #cv2.imshow('fff',Input_img)
        
        #img= cv2.resize(img,(256,256), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(cv2.resize(img,(150,150)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((150,150)))
        #cv2.imshow('fff',render)
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=250, y=50)
##        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
##        image2.image = render
##        image2.place(x=250, y=50)
        #cv2.destroyAllWindows()
        contents="Image Loadeded successfully !!"
        
        T = Text(self, height=21, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        self.Input_img=Input_img
    def close_window(): 
        Window.destroy()
    def preprocess(self, event=None):
        global T,rep
        contents="Pre-Processing ..."
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(256,256))
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray= median(gray)
        gray= cv2.medianBlur(img,5)
        img= cv2.resize(gray,(256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite('pre.png',img)
        #self.from_array = Image.fromarray(cv2.resize(img,(150,150)))
        #render = ImageTk.PhotoImage(self.from_array)
        load = Image.open('pre.png')
        render = ImageTk.PhotoImage(load.resize((150,150)))
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=500, y=50)
        
                
        contents="Pre-Processing completed successfully using Median filter  "
                   
        T = Text(self, height=20, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
    def segment(self, event=None):
        contents ="Segmentation Processing..."
        global T,rep,segmented_image
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        img_org = cv2.imread(rep[0])
        image=cv2.resize(img_org,(256,256))

        #cv2.imshow('Original Image',image)

        median = cv2.medianBlur(image,5)
        #cv2.imshow('Median Filtered Image',median)
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_val = (0,60,0)
        high_val = (179,255,255)
        # Threshold the HSV image 
        mask = cv2.inRange(image, low_val,high_val)
        # remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
        # apply mask to original image
        result = cv2.bitwise_and(image, image,mask=mask)

        #show image
        #cv2.imshow("Result", result)
        #cv2.imshow("Mask", mask)
        self.from_array = Image.fromarray(cv2.resize(result,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image5.image = render
        image5.place(x=250, y=270)

        self.from_array = Image.fromarray(cv2.resize(mask,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image4.image = render
        image4.place(x=750, y=50)
        #cv2.destroyAllWindows()
        contents="Segmentation completed successfully !!  Thresholding Otsu's method  \n "
        
        T = Text(self, height=20, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        
    def feature(self, event=None):
        contents ="Feature Extracting..."
        global T,rep,xname
        global segmented_image,X
        

        outputVectors = []
        loadedImages = []

        input1=np.zeros([20, 1])
        target=[]
        main_img = cv2.imread(rep[0])


        # Image operation using thresholding 
        image= cv2.imread(rep[0])
        image=cv2.resize(image,(256,256))

        #cv2.imshow('Original Image',image)

        median = cv2.medianBlur(image,5)
        #cv2.imshow('Median Filtered Image',median)

        img = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        def filter_image(img,mask):
            r = image[:,:,0]*mask
            g = image[:,:,1]*mask
            b = image[:,:,2]*mask
            return np.dstack([r,g,b])
        img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('Semented Image',thresh1)

        res=thresh1

        z=texture(res)
##        tar=0
##        for folder in folder_list:
##        
##            # create a path to the folder
##            path = 'dataset/' + str(folder)
##            img_files = os.listdir(path)
##            print(path)
##            for file in img_files:
##                src = os.path.join(path, file)
##                main_img = cv2.imread(src)
##                res=preprocess(main_img)
##                y=segmentation(res)
##                z=texture(y)
##                if len(z)==20:
##                    input1= np.c_[input1,z]
####                    target.append(tar)
####        tar=tar+1
        print(z)
        contents="Feature Extraction completed successfully !!"
        T = Text(self, height=19, width=25)
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)

    def classification(self, event=None):
        outputVectors = []
        loadedImages = []
        input1=np.zeros([20, 1])
        target=[]
        main_img = cv2.imread(rep[0])
        


        # Image operation using thresholding 
        image= cv2.imread(rep[0])
        image=cv2.resize(image,(256,256))

        #cv2.imshow('Original Image',image)

        median = cv2.medianBlur(image,5)
        #cv2.imshow('Median Filtered Image',median)

        img = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        def filter_image(img,mask):
            r = image[:,:,0]*mask
            g = image[:,:,1]*mask
            b = image[:,:,2]*mask
            return np.dstack([r,g,b])
        img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('Semented Image',thresh1)

        res=thresh1

        z=texture(res)


        
        #res=preprocess(main_img)
        #y=segmentation(res)
        #z=texture(main_img)
        if len(z)==20:
            input1= np.c_[input1,z]  
        clas=["diseased","normal"]
        file = "finalized_model_DT.sav"
        clf_model = pickle.load(open(file, 'rb'))
        pred=clf_model.predict(z.reshape(1,20))
        print((pred))
        res = clas[int(pred)]
        print('Given Arecanut is : '+res)
        contents="classification completed successfully!!"
        messagebox.showinfo('Given Arecanut is: ',res)

        T = Text(self, height=19, width=25)
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)


root = Tk()
root.geometry("1200x720")
app = Window(root)
root.mainloop()

        
