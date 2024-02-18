import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from google.colab import files
import os
import zipfile
from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from tkinter import filedialog
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import cv2
from matplotlib import pyplot as plt
from tkinter import filedialog
import math
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
from PIL import Image,ImageStat
import matplotlib.image as mpimg



def reshape_list(arr):
	return np.reshape(arr, arr.size)

# median filter
def preprocess(image):
    image=cv2.resize(image,(256,256))
    #cv2.imshow('Original Image',image)
    median = cv2.medianBlur(image,5)
    #cv2.imshow('Median Filtered Image',median)
    #cv2.imwrite("median.jpg", median)
    return median
   
###ostu segmentation
def segmentation(median):
    img = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
##    plt.axis('off')
##    plt.imshow(img)
    def filter_image(img,mask):
        r = image[:,:,0]*mask
        g = image[:,:,1]*mask
        b = image[:,:,2]*mask
        return np.dstack([r,g,b])
    img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Otsu Threshold', thresh1)
    #cv2.imwrite("otsu.jpg",thresh1)
    return thresh1

# GLCM properties
def contrast_feature(matrix_coocurrence):
    contrast = graycoprops(matrix_coocurrence, 'contrast')
    return contrast
def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity
def homogeneity_feature(matrix_coocurrence):
    homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity
def energy_feature(matrix_coocurrence):
    energy = graycoprops(matrix_coocurrence, 'energy')
    return energy
def correlation_feature(matrix_coocurrence):
    correlation = graycoprops(matrix_coocurrence, 'correlation')
    return correlation



def texture(img):
    ##feature extraction on GLCM
    PATCH_SIZE = 20
    training_length = 200
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/2]
    np.set_printoptions(precision=10)

##    gray = color.rgb2gray(img)
    image = img_as_ubyte(img)
    ##io.imshow(image)
    matrix_texture=[]
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/4, 3*np.pi/4], levels=max_value)
    
    z=contrast_feature(matrix_coocurrence)
    z=z.reshape(4,)
    z1=dissimilarity_feature(matrix_coocurrence)
    z1=z1.reshape(4,)
    z2=homogeneity_feature(matrix_coocurrence)
    z2=z2.reshape(4,)
    z3=energy_feature(matrix_coocurrence)
    z3=z3.reshape(4,)
    z4=correlation_feature(matrix_coocurrence)
    z4=z4.reshape(4,)
    
##    matrix_texture=np.c_[matrix_texture,]
##    
##    matrix_texture=np.c_[matrix_texture,]
##    matrix_texture=np.c_[matrix_texture,]
##    matrix_texture=np.c_[matrix_texture,]
##    matrix_texture=np.c_[matrix_texture,]

    texture = np.concatenate((z,z1, z2,z3,z4), axis=0)
    return texture 




  
