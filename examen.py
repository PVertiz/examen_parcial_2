import numpy as np
import cv2
import random
import math
import sys

def producto_punto(vector1, vector2):
    return np.array(vector1)*np.array(vector2)

def producto_cruz(vector1, vector2):
    return np.cross(np.array(vector1), np.array(vector2))

def distancia_2_puntos(punto1,punto2):
    return(np.sqrt((punto1[0]-punto2[0])**2 + (punto1[1]-punto2[1])**2)) 

def interseccion(ecuacion1, ecuacion2): #mandar vector con indicies de x+y=A 
    ecuaciones=[]
    c=[]
    ecuaciones.append(ecuacion1[0:-1])
    ecuaciones.append(ecuacion2[0:-1])
    c.append(ecuacion1[-1])
    c.append(ecuacion2[-1])
    
    return np.linalg.inv(ecuaciones).dot(c) #retorna nuevo vector con las coordenadas de la interseccion 

imagen1 = cv2.imread('Jit1_peque.jpg')
imagen=cv2.GaussianBlur(imagen1, (7, 7), 0)
RGB_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)# convertir de bgr a rgb 


print(RGB_img)

(filas, columnas) = RGB_img.shape[0:2]

def inicializar_centroides(k):
    centroides=[]
    #creamos centroides aleatorios
    for i in range(k):
        coordenadas=[]
        for j in range(3):
            coordenadas.append(random.randrange(0, 255))
        centroides.append(coordenadas)

    return centroides

#recivimos 2 puntos y regresamos la distancia entre ellos
def distancia_euclidiana(punto1,punto2):
    suma_aux = 0 #suma auxiliar
    for i in range(len(punto1)): 
        suma_aux += math.pow(punto1[i]-punto2[i], 2)
  
    #regresamos la raiz cuadrada
    return math.sqrt(suma_aux)

#actualizamos centroides
def actualizar_centroides(centroides,agrupaciones):
    nuevos_centroides=[]
    for i in range(len(centroides)):
        centroide_aux=[]
        x_aux=0
        y_aux=0
        z_aux=0
        for j in range(len(agrupaciones[i])):
            x_aux+= agrupaciones[i][j][0]
            y_aux+= agrupaciones[i][j][1]
            z_aux+= agrupaciones[i][j][2]
        #print(agrupaciones[i])    
        #print(len(agrupaciones[i]))
        divisor=len(agrupaciones[i])
        #en caso de que el divisor sea igual a cero
        if (divisor==0):
            divisor=1
        x_aux=x_aux/divisor
        y_aux=y_aux/divisor
        z_aux=z_aux/divisor

        centroide_aux.append(x_aux)
        centroide_aux.append(y_aux)
        centroide_aux.append(z_aux)
        
        nuevos_centroides.append(centroide_aux)
    return nuevos_centroides
        
def clasificar(centroides,pixel):
    minimo = 100000
    index = -1

    for i in range(len(centroides)):
        distancia=distancia_euclidiana(centroides[i],pixel)
        if (distancia < minimo):#establecemos cual es el centroide más cercano 
            minimo = distancia
            index = i
    return index

#separamos la imagen de acuerdo a los centroides
def segmentar_imagen(imagen,k,centroides):
    imagen_segmentada=np.zeros([filas, columnas, 3], dtype=np.uint8)
    for w in range(k): 
        valor_aux=centroides[w]
        for i in range(filas):
            for j in range(columnas):
                if(imagen[i][j]==w):
                    imagen_segmentada[i][j]=valor_aux
    return imagen_segmentada
#separamos la imagen en blanco y negro con solo los jitomates
def crear_imagen_con_objetos(imagen,centroide):
    imagen_segmentada=np.zeros([filas, columnas, 1], dtype=np.uint8)
    
    for i in range(filas):
        for j in range(columnas):
            if(imagen[i][j]==centroide):
                imagen_segmentada[i][j]=[255]
            else:
                imagen_segmentada[i][j]=[0]
    return imagen_segmentada

#obtenemos los puntos mas alejados de los bordes
def obtener_diametro(bordes):
    distancia_minima=-100
    extremos=[]
    coordenadas=[]
    distancia_temp=0
    for i in range(len(bordes)):
        for j in range(len(bordes)):
           
            distancia_temp=distancia_euclidiana(bordes[i][0],bordes[j][0])
            if(distancia_temp>distancia_minima):
                distancia_minima=distancia_temp
                punto1=bordes[i][0]
                punto2=bordes[j][0]
    
    extremos.append(punto1)
    extremos.append(punto2)
    return extremos

def calcular_kmeans(k,imagen,iteraciones):
    centroides=inicializar_centroides(k)
    
    #print(agrupacion_de_pixeles)
    

    for w in range(iteraciones):
        print("ciclo "+str(w))
        agrupacion_de_pixeles=[[] for j in range(k)]
        imagen_pixeles_agrupados=np.zeros([filas, columnas, 1], dtype=np.uint8)
        for i in range(filas):
            for j in range(columnas):
                val_aux=imagen[i][j]
                #print(val_aux)
                grupo_aux=clasificar(centroides,imagen[i][j])
                imagen_pixeles_agrupados[i][j]=grupo_aux
                #print(val_aux)
                agrupacion_de_pixeles[grupo_aux].append(val_aux)
                #print(agrupacion_de_pixeles) 
        centroides_temp=centroides
        #print(imagen_pixeles_agrupados)
        #print("bandera 1")
        # for i in range(len(agrupacion_de_pixeles)):
        #     print(agrupacion_de_pixeles[i])
        
        nuevo_grupo=[]
        #print(list(agrupacion_de_pixeles[0][0]))
        for i in range(len(agrupacion_de_pixeles)):
            agrupacion_aux=[]
            for j in range(len(agrupacion_de_pixeles[i])):
                agrupacion_aux.append(list(agrupacion_de_pixeles[i][j]))
            nuevo_grupo.append(agrupacion_aux)

        #print(nuevo_grupo)
        centroides=actualizar_centroides(centroides,nuevo_grupo)
        
        
        if(centroides_temp==centroides):
            break
    agrupador_objetos=0
    #verificar que centroide es el que tiene los circulos
    for i in range(len(centroides)):
            if(abs(centroides[i][0]-centroides[i][1])>50 and abs(centroides[i][0]-centroides[i][2])>50):
                agrupador_objetos=i
    



    imagen_segmentada=segmentar_imagen(imagen_pixeles_agrupados,k,centroides)
    imagen_con_bordes=imagen_segmentada
    imagen_objetos=crear_imagen_con_objetos(imagen_pixeles_agrupados,agrupador_objetos)
    ret, thresh = cv2.threshold(imagen_objetos, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numero_figuras=len(contours)
    print(len(contours))
    lineas=[]
    for i in range(numero_figuras):
        lineas.append(obtener_diametro(contours[i]))
        
    #print(lineas)
    imagen_copia=imagen1
    print("distancias: ")
    for i in range(numero_figuras):
        print(distancia_euclidiana(lineas[i][0],lineas[i][1]))
        imagen_copia = cv2.line(imagen_copia, lineas[i][0], lineas[i][1],(249, 251, 26), 3)

    cv2.drawContours(imagen_con_bordes, contours, -1, (0,255,0), 3)    
    #cv2.imshow("imagen segmentada",imagen_segmentada)
    cv2.imshow("imagen con objetos",imagen_objetos)
    cv2.imshow("imagen con bordes",imagen_con_bordes)
    cv2.imshow("imagen con trazos",imagen_copia)
    cv2.imwrite('segmantada.jpg', imagen_segmentada)
    
    cv2.waitKey(0)





    



calcular_kmeans(4,RGB_img,20)    


#print(interseccion([1,-1,10], [2,1,5]))


cv2.destroyAllWindows()
#https://www.coursera.org/learn/procesamiento-de-imagenes#syllabus
#https://www.geeksforgeeks.org/k-means-clustering-introduction/
#contours moments
# finf contours