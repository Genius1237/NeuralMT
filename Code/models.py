import numpy as np
from keras.layers import Embedding
from keras.models import Sequential

model=Sequential()
model.add(Embedding(1000,10,input_length=1))

#input_array=np.random.randint(1000,size=(32,10))
input_array=np.arange(0,999,dtype=float)
print(input_array.shape)
#print(input_array)

model.compile('rmsprop','mse')
output_array=model.predict(input_array)
print(np.linalg.norm(output_array[0]+output_array[1]))
print(np.linalg.normt(output_array[2]))