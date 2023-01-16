import pandas as pd
import urllib.request

registro_csv = pd.read_csv("instagram_scrapeado.csv")
nuevos_registros = []

for registro in registro_csv.values:
    url_imagen = registro[4]
    cantidad_likes = registro[8]

    #para que no se eliminen fotos:
    username = registro[0]
    unique_id = url_imagen[-23:-14]

    urllib.request.urlretrieve(url_imagen,f"imagenes/{cantidad_likes}_{username}_{unique_id}.jpg")

