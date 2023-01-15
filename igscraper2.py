#Este ig scraper solo recolecta imagen y likes
# _ac2d es de las imagenes (cuando pasas el mouse encima)
# _aagu son las imagenes (12 x ite)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
import time
from bs4 import BeautifulSoup
import urllib.request
import requests
import ssl
from dotenv import load_dotenv
import os
import re
import emoji
import pandas as pd

load_dotenv()
# Variables de entorno
INSTAGRAM_USER = os.getenv("INSTAGRAM_USER")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")

PATH = str(os.getcwd()) + "/chromedriver" #CUando es windows se usa .exe, sino sin .nada
driver = webdriver.Chrome(PATH)

def bajar(driver):
    cont = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while cont<3: #maso mil imagenes es 30 #corre 4 veces si dice 3

        #POR LO QUE VEO SE CARGAN DE 12 EN 12 LAS IMAGENES EN INSTAGRAM
        # Scroll down to the bottom.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load the page.
        time.sleep(2.3)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break

        last_height = new_height
        cont = cont + 1


def movimiento_mouse(driver):
    mouse_mov = driver.find_elements(By.CLASS_NAME, '_aabd')
    movimiento = ActionChains(driver)
    print("Total real imagenes: ", (3+1)*12)
    print("TAMANIO MAXIMO = ",len(mouse_mov)) #45 con 4,6 osea máximo 45 (3.75)
    cont = 0
    for i in mouse_mov:
        #Con 2 y 1.5 funciona. 
        # Si quito el primero y el 2do es 2, no funciona
        dataset = {}

        movimiento = ActionChains(driver)
        time.sleep(2) 
        movimiento.move_to_element(i).perform()
        time.sleep(1.5) 

        #Likes y comentarios
        like_comentario = driver.find_elements(By.CLASS_NAME, '_aad3')
        like = like_comentario[0]
        coments = like_comentario[1]
        #likes.append(like.text)
        print(cont,": ", like.text, coments.text)

        #Para determinar url:
        inner = i.get_attribute('innerHTML')
        if 'svg' not in inner:
            enlace = re.findall(r'href="(.*?)"', inner)
            enlace = "https://www.instagram.com/" + enlace[0]
            #posts.append(enlace)

            dataset[enlace] = like.text

        print("################")
        cont = cont + 1
    
    return dataset


def scraping(usernames):
    #ACA DEBERIA IR LOGIN PERO ME SALTO ESA PARTE
    driver.get("https://www.instagram.com/")
    ssl._create_default_https_context = ssl._create_unverified_context
    time.sleep(4)
    username = driver.find_element("css selector", "input[name='username']")
    password = driver.find_element("css selector", "input[name='password']")
    username.clear()
    password.clear()
    username.send_keys(INSTAGRAM_USER)
    password.send_keys(INSTAGRAM_PASSWORD)
    login = driver.find_element("css selector", "button[type='submit']").click()
    time.sleep(7)

    for user in usernames:
        driver.get(f"https://www.instagram.com/{user}")

        time.sleep(5)

        #######################################
        informacion_usuario = driver.find_elements(By.CLASS_NAME, '_ac2a')
        #publicaciones 0, seguidores 1, seguidos 2
        publis = informacion_usuario[0].get_attribute('innerHTML')
        cantidad_publicaciones = int("".join(re.findall(r'\d+', publis)))

        seguidores = informacion_usuario[1].get_attribute('title')
        cantidad_seguidores = int("".join(re.findall(r'\d+', seguidores)))

        seguidos = informacion_usuario[2].get_attribute('innerHTML')
        cantidad_seguidos = int("".join(re.findall(r'\d+', seguidos)))
        ######################################            

        veces = 3 #Con veces de 3 se puede ver máximo de imagenes (48, hay 3 que no se ven)
        dataset = {}

        for i in range(0,2): #1 sola iteración de 45 imagenes -> 45*5 = 225
            bajar(driver)
            dataset2 = movimiento_mouse(driver)
            dataset = dict(**dataset, **dataset2)
            time.sleep(2)   

        driver.close()  
        '''print("\n\n\nVerificar si pertenecen")
        print(likes)
        print(posts)
        print("Largo likes: ", len(likes))
        print("Largo posts: ", len(posts))'''
        print("#######\nDataset final: ", dataset)
        print(len(dataset))
        
        #
        # Falta agregar una forma de leer la tabla que ya está lista para concatenar
        #
        
        dataset_en_csv = []

        for i in dataset:
            dataset_en_csv.append([i, dataset.get(i)])
            
        df = pd.DataFrame(dataset_en_csv, columns=['url_imagen', 'cantidad_likes'])
        df.to_csv("ig_scrapeado_2.csv")
        

usernames = ["centecproccs"]
scraping(usernames)
 