# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

# %% Regresion logistica con el dataset digits
# =============================================================================

# El conjunto de datos digits es uno de los conjuntos de datos que incluye 
# scikit-learn y que no requieren la descarga de ningún archivo de algún 
# sitio web externo. 


#%%  Cargamos modulos y datos 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()


# %% Veamos como cargo los datos 

print(dir(digits))

# Descripcion del dataset
print(digits["DESCR"])

# data 
print(digits['data'])
print(digits['data'].shape)
print(digits['data'][0])
print(digits['data'][0].shape)

# Otros atributos de digits
print(digits['images'].shape)
print(digits['images'][0])
print(digits['images'][0].shape)

print(digits['target'][0])
print(digits['target_names'][0])


# %% Mostrar las imágenes y las etiquetas (conjunto de datos de dígitos)



import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
 

#%%  split train 90% & test 10%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# %% Construccion del modelo 

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(max_iter= 3000)

logisticRegr.fit(x_train, y_train)

# %% Predecir etiquetas para nuevos datos (nuevas imágenes)


# REtorna un numpy array
# prediccion para una  observacion 
logisticRegr.predict(x_test[0].reshape(1,-1))


# %% Predecir para múltiples observaciones (imágenes) a la vez

logisticRegr.predict(x_test[0:10])

# %% Haga predicciones sobre datos de prueba completos

predictions = logisticRegr.predict(x_test)

# %% Medición del rendimiento/performnce del modelo 
# (conjunto de datos digits)

# Si bien hay otras formas de medir el rendimiento del modelo (precisión, 
# recall, Score F1, curva ROC, etc.), mantendremos esto simple y usaremos 
# la precisión/accuracy como nuestra métrica. Para hacer esto, veremos cómo 
# funciona el modelo con los nuevos datos (conjunto de prueba)

# Para hacer esto, veremos cómo funciona el modelo con los nuevos datos 
# (conjunto de prueba)

# la precisión se define como:
    # (fracción de predicciones correctas): predicciones correctas / número total de puntos de datos
# 
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)
# Nuestra precisión fue del 95,3%.

# %% Matriz de confusión (conjunto de datos digits)

# Una matriz de confusión es una tabla que se utiliza a menudo para 
# describir el desempeño de un modelo de clasificación (o "clasificador") 
# en un conjunto de datos de prueba para los que se conocen los valores 
# verdaderos. En esta sección, solo estoy mostrando dos paquetes de Python 
# (Seaborn y Matplotlib) para hacer que las matrices de confusión sean más 
# comprensibles y visualmente atractivas.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# La matriz de confusión a continuación no es visualmente súper informativa ni 
# visualmente atractiva.
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Método 1 (Seaborn)
# Como puede ver a continuación, este método produce una matriz de 
# confusión más comprensible y visualmente legible utilizando seaborn.
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# Método 2 (Matplotlib) 
# Este método es claramente mucho más código. Solo quería mostrarle a la 
# gente cómo hacerlo en matplotlib también.
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
                     horizontalalignment='center',
                     verticalalignment='center')
 
#                    
# Esto es un cambio que hicimos en la rama Alumno1
#
