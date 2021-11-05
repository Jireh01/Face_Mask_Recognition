import cv2
import os
import numpy as np
#Colocar el path del dataset de Kagle.
#Path = direccion de donde se encuentra el dataset
path = " "
carpetas = os.listdir(path)

labels = []
arrayImagenes = []
aux = 0
for i in carpetas:
     path_imga_indi = path + "/" + i
     
     for j in os.listdir(path_imga_indi):
          image_path = path_imga_indi + "/" + j
          print(image_path)
          image = cv2.imread(image_path, 0)
          cv2.imshow("Image", image)
          cv2.waitKey(10)
          arrayImagenes.append(image)
          labels.append(aux)
     aux += 1
print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1))
# LBPH FaceRecognizer
face_mask = cv2.face.LBPHFaceRecognizer_create()
# Entrenamiento
print("Entrenando...")
face_mask.train(arrayImagenes, np.array(labels))
# Almacenar modelo
face_mask.write("train_model.xml")
print("Modelo almacenado")
