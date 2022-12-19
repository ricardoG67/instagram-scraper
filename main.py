from unicodedata import name
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import requests
import ssl
from dotenv import load_dotenv
import os
import re

#https://www.youtube.com/watch?v=-3hCfDRkq9s&ab_channel=Cryproot

load_dotenv()
# Variables de entorno
INSTAGRAM_USER = os.getenv("INSTAGRAM_USER")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")

PATH = str(os.getcwd()) + "/chromedriver" #CUando es windows se usa .exe, sino sin .nada
driver = webdriver.Chrome(PATH)

#ACA DEBERIA IR LOGIN PERO ME SALTO ESA PARTE
driver.get("https://www.instagram.com/bellapoarch")
ssl._create_default_https_context = ssl._create_unverified_context

time.sleep(3)

posts = []
links = driver.find_elements("tag name", "a")
for link in links:
    post = link.get_attribute('href')
    if '/p/' in post:
        posts.append(post) #Sin login son 12 imagenes

#Si es carrusel, tendrá un div con class  "_9zm2"

for post in posts:
    driver.get(post)
    time.sleep(3)
    
    if len(driver.find_elements("tag name", "video")) == 0 and len(driver.find_elements(By.CLASS_NAME, '_9zm2'))==0:
        #Lo que está aca no es ni video ni carrusel
        url_imagen = driver.find_element(By.CLASS_NAME, 'xu96u03').get_attribute('src')
        print(url_imagen)    
        #https://stackoverflow.com/questions/7263824/get-html-source-of-webelement-in-selenium-webdriver-using-python
        likes = driver.find_elements(By.CLASS_NAME, '_aada')[-1].get_attribute('innerHTML')

        likes = int("".join(re.findall(r'\d+', likes))) #obtiene los numeros en lista, los pasa a string  y luego a int
        print(likes)
        
driver.close()