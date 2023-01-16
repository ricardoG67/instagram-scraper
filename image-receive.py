import cv2 as cv
import re
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import urllib.request
import numpy as np
import os

#https://thecleverprogrammer.com/2021/06/22/r2-score-in-machine-learning/
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

def extractor(name_imagen):

    img_original = cv.imread(f"imagenes/{name_imagen}")
    vertical = (img_original.shape)[0]
    vertical = int(vertical)

    horizontal = (img_original.shape)[1]
    horizontal = int(horizontal)

    #Puntos
    vertical_p1 = vertical/5
    vertical_p2 = (2*vertical)/5
    vertical_p3 = (3*vertical)/5
    vertical_p4 = (4*vertical)/5


    horizontal_p1 = (horizontal)/5
    horizontal_p2 = (2*horizontal)/5
    horizontal_p3 = (3*horizontal)/5
    horizontal_p4 = (4*horizontal)/5


    #Cuadrante1:
    cuadrante_1 = img_original[0:int(vertical_p1), 0:int(horizontal_p1)]

    #Cuadrante2:
    cuadrante_2 = img_original[0:int(vertical_p1), int(horizontal_p1):int(horizontal_p2)]

    #Cuadrante3:
    cuadrante_3 = img_original[0:int(vertical_p1), int(horizontal_p2):int(horizontal_p3)]

    #Cuadrante4:
    cuadrante_4 = img_original[0:int(vertical_p1), int(horizontal_p3):int(horizontal_p4)]

    #Cuadrante5:
    cuadrante_5 = img_original[0:int(vertical_p1), int(horizontal_p4):int(horizontal)]


    #########

    #Cuadrante6:
    cuadrante_6 = img_original[int(vertical_p1):int(vertical_p2), 0:int(horizontal_p1)]

    #Cuadrante7:
    cuadrante_7 = img_original[int(vertical_p1):int(vertical_p2), int(horizontal_p1):int(horizontal_p2)]

    #Cuadrante8:
    cuadrante_8 = img_original[int(vertical_p1):int(vertical_p2), int(horizontal_p2):int(horizontal_p3)]

    #Cuadrante9:
    cuadrante_9 = img_original[int(vertical_p1):int(vertical_p2), int(horizontal_p3):int(horizontal_p4)]

    #Cuadrante10:
    cuadrante_10 = img_original[int(vertical_p1):int(vertical_p2), int(horizontal_p4):int(horizontal)]

    #########

    #Cuadrante11:
    cuadrante_11 = img_original[int(vertical_p2):int(vertical_p3), 0:int(horizontal_p1)]

    #Cuadrante12:
    cuadrante_12 = img_original[int(vertical_p2):int(vertical_p3), int(horizontal_p1):int(horizontal_p2)]

    #Cuadrante13:
    cuadrante_13 = img_original[int(vertical_p2):int(vertical_p3), int(horizontal_p2):int(horizontal_p3)]

    #Cuadrante14:
    cuadrante_14 = img_original[int(vertical_p2):int(vertical_p3), int(horizontal_p3):int(horizontal_p4)]

    #Cuadrante15:
    cuadrante_15 = img_original[int(vertical_p2):int(vertical_p3), int(horizontal_p4):int(horizontal)]

    #########

    #Cuadrante16:
    cuadrante_16 = img_original[int(vertical_p3):int(vertical_p4), 0:int(horizontal_p1)]

    #Cuadrante17:
    cuadrante_17 = img_original[int(vertical_p3):int(vertical_p4), int(horizontal_p1):int(horizontal_p2)]

    #Cuadrante18:
    cuadrante_18 = img_original[int(vertical_p3):int(vertical_p4), int(horizontal_p2):int(horizontal_p3)]

    #Cuadrante19:
    cuadrante_19 = img_original[int(vertical_p3):int(vertical_p4), int(horizontal_p3):int(horizontal_p4)]

    #Cuadrante20:
    cuadrante_20 = img_original[int(vertical_p3):int(vertical_p4), int(horizontal_p4):int(horizontal)]

    #########

    #Cuadrante21:
    cuadrante_21 = img_original[int(vertical_p4):(vertical), 0:int(horizontal_p1)]

    #Cuadrante22:
    cuadrante_22 = img_original[int(vertical_p4):(vertical), int(horizontal_p1):int(horizontal_p2)]

    #Cuadrante23:
    cuadrante_23 = img_original[int(vertical_p4):(vertical), int(horizontal_p2):int(horizontal_p3)]

    #Cuadrante24:
    cuadrante_24 = img_original[int(vertical_p4):(vertical), int(horizontal_p3):int(horizontal_p4)]

    #Cuadrante25:
    cuadrante_25 = img_original[int(vertical_p4):(vertical), int(horizontal_p4):int(horizontal)]

    cuadrantes = [cuadrante_1, cuadrante_2 ,cuadrante_3, cuadrante_4,cuadrante_5,
                  cuadrante_6, cuadrante_7, cuadrante_8,cuadrante_9, cuadrante_10,
                   cuadrante_11, cuadrante_12, cuadrante_13, cuadrante_14, cuadrante_15, 
                  cuadrante_16, cuadrante_17,cuadrante_18,cuadrante_19,cuadrante_20,
                  cuadrante_21,cuadrante_22,cuadrante_23,cuadrante_24,cuadrante_25]
                  
    ###################

    keypoints = []
    for i in cuadrantes:
        
        gray= cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)

        keypoints.append(len(kp))

    
    kp_fast = []
    for i in cuadrantes:
        
        gray = cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(gray,None)

        kp_fast.append(len(kp))

    kp_orb = []
    for i in cuadrantes:
        
        gray = cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        kp = orb.detect(gray,None)

        kp_orb.append(len(kp))

    keypoints = np.array(keypoints)
    kp_fast = np.array(kp_fast)
    kp_orb = np.array(kp_orb)

    return keypoints, kp_fast, kp_orb

registro_csv = pd.read_csv("instagram_scrapeado.csv")
para_predecir = []

for name_imagen in os.listdir("imagenes"):

    cantidad_likes = re.findall(r'[\d]+\_',(name_imagen))
    cantidad_likes = cantidad_likes[0][:-1] #eliminando _ y como lo da en lista obtengo el 1er valor

    sift_25_kp, fast_25_kp, orb_25_fast = extractor(name_imagen)

    c1 = sift_25_kp[0]
    c2 = sift_25_kp[1]
    c3 = sift_25_kp[2]
    c4 = sift_25_kp[3]
    c5 = sift_25_kp[4]
    c6 = sift_25_kp[5]
    c7 = sift_25_kp[6]
    c8 = sift_25_kp[7]
    c9 = sift_25_kp[8]
    c10 = sift_25_kp[9]
    c11 = sift_25_kp[10]
    c12 = sift_25_kp[11]
    c13 = sift_25_kp[12]
    c14= sift_25_kp[13]
    c15 = sift_25_kp[14]
    c16= sift_25_kp[15]
    c17= sift_25_kp[16]
    c18= sift_25_kp[17]
    c19= sift_25_kp[18]
    c20= sift_25_kp[19]
    c21= sift_25_kp[20]
    c22= sift_25_kp[21]
    c23= sift_25_kp[22]
    c24= sift_25_kp[23]
    c25= sift_25_kp[24]

    c1_fast = fast_25_kp[0]
    c2_fast = fast_25_kp[1]
    c3_fast = fast_25_kp[2]
    c4_fast = fast_25_kp[3]
    c5_fast = fast_25_kp[4]
    c6_fast = fast_25_kp[5]
    c7_fast = fast_25_kp[6]
    c8_fast = fast_25_kp[7]
    c9_fast = fast_25_kp[8]
    c10_fast = fast_25_kp[9]
    c11_fast = fast_25_kp[10]
    c12_fast = fast_25_kp[11]
    c13_fast = fast_25_kp[12]
    c14_fast = fast_25_kp[13]
    c15_fast = fast_25_kp[14]
    c16_fast = fast_25_kp[15]
    c17_fast = fast_25_kp[16]
    c18_fast = fast_25_kp[17]
    c19_fast = fast_25_kp[18]
    c20_fast = fast_25_kp[19]
    c21_fast = fast_25_kp[20]
    c22_fast = fast_25_kp[21]
    c23_fast = fast_25_kp[22]
    c24_fast = fast_25_kp[23]
    c25_fast = fast_25_kp[24]

    c1_orb = orb_25_fast[0]
    c2_orb = orb_25_fast[1]
    c3_orb = orb_25_fast[2]
    c4_orb = orb_25_fast[3]
    c5_orb = orb_25_fast[4]
    c6_orb = orb_25_fast[5]
    c7_orb = orb_25_fast[6]
    c8_orb = orb_25_fast[7]
    c9_orb = orb_25_fast[8]
    c10_orb = orb_25_fast[9]
    c11_orb = orb_25_fast[10]
    c12_orb = orb_25_fast[11]
    c13_orb = orb_25_fast[12]
    c14_orb= orb_25_fast[13]
    c15_orb = orb_25_fast[14]
    c16_orb= orb_25_fast[15]
    c17_orb= orb_25_fast[16] 
    c18_orb= orb_25_fast[17]
    c19_orb= orb_25_fast[18]
    c20_orb= orb_25_fast[19]
    c21_orb= orb_25_fast[20]
    c22_orb= orb_25_fast[21]
    c23_orb= orb_25_fast[22]
    c24_orb= orb_25_fast[23]
    c25_orb= orb_25_fast[24]

    para_predecir.append([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,
    c1_fast,c2_fast,c3_fast, c4_fast, c5_fast, c6_fast, c7_fast, c8_fast, c9_fast, c10_fast,c11_fast,c12_fast, 
    c13_fast, c14_fast, c15_fast, c16_fast,c17_fast,c18_fast,c19_fast, c20_fast, c21_fast, c22_fast, c23_fast, c24_fast, c25_fast,
    c1_orb,c2_orb,c3_orb, c4_orb, c5_orb, c6_orb, c7_orb, c8_orb, c9_orb, c10_orb,c11_orb,c12_orb, c13_orb, c14_orb, c15_orb, c16_orb,
    c17_orb,c18_orb,c19_orb, c20_orb, c21_orb, c22_orb, c23_orb, c24_orb, c25_orb,cantidad_likes]) 

df = pd.DataFrame(para_predecir, columns=["c1_sift", "c2_sift", "c3_sift", "c4_sift", "c5_sift", "c6_sift", "c7_sift", "c8_sift", "c9_sift",
                                  "c10_sift", "c11_sift", "c12_sift", "c13_sift", "c14_sift", "c15_sift", "c16_sift",
                                  "c17_sift", "c18_sift", "c19_sift", "c20_sift", "c21_sift", "c22_sift", "c23_sift", "c24_sift", "c25_sift",

                                  "c1_fast","c2_fast","c3_fast","c4_fast","c5_fast","c6_fast","c7_fast","c8_fast","c9_fast",
                                  "c10_fast","c11_fast","c12_fast","c13_fast","c14_fast","c15_fast","c16_fast",
                                  "c17_fast","c18_fast","c19_fast","c20_fast","c21_fast","c22_fast","c23_fast","c24_fast","c25_fast",

                                  "c1_orb","c2_orb","c3_orb","c4_orb","c5_orb","c6_orb","c7_orb","c8_orb","c9_orb",
                                  "c10_orb","c11_orb","c12_orb","c13_orb","c14_orb","c15_orb","c16_orb",
                                  "c17_orb","c18_orb","c19_orb","c20_orb","c21_orb","c22_orb","c23_orb","c24_orb","c25_orb","target"])

print(df)
print("\n\n\n\n")

x = df.values[:,:-1]
y = df.target.values
escalador = StandardScaler()
x_norm = escalador.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2)
#x_train.shape, y_train.shape, x_test.shape, y_test.shape, y.shape

from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score

classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

cont = 0
accuracy = 0
indice = None
for item in classifiers:
    print(item)
    clf = item
    clf.fit(x_train, y_train)
    print("porcentaje acuraccy:", clf.score(x_test, y_test))
    valor_r = cross_val_score(item, x_test, y_test, cv=5).mean()
    print(f"cross_val: ", valor_r) #Ver para cambiar esto
    if valor_r >= accuracy:
        accuracy = valor_r
        indice = cont
    cont = cont + 1
    print("\n####################\n")

print("###########################\n")
print("###########################\n")
print("###########################\n")
print(f"Mayor r² es {accuracy}")