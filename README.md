# Advanced Topics in Deep Learning (AIDL_B02)
## Liakopoulos Alexandros
## mscaidl-0007

# Anime-Faces-and-Names
* Dataset found on Kaggle but after a while the owner removed it. So, there is no link to refer to.
* ### Porpose of the dataset is to create new *Anime Characcters* by using Deep Convolutional Generative Adversarial Network to generate new images.

# Model Review
* # Generator
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_4 (Dense)             (None, 345600)            34560000  
                                                                 
 leaky_re_lu_11 (LeakyReLU)  (None, 345600)            0         
                                                                 
 reshape_1 (Reshape)         (None, 45, 30, 256)       0         
                                                                 
 conv2d_transpose_5 (Conv2DT  (None, 45, 30, 256)      1638400   
 ranspose)                                                       
                                                                 
 batch_normalization_8 (Batc  (None, 45, 30, 256)      1024      
 hNormalization)                                                 
                                                                 
 leaky_re_lu_12 (LeakyReLU)  (None, 45, 30, 256)       0         
                                                                 
 conv2d_transpose_6 (Conv2DT  (None, 90, 60, 128)      819200    
 ranspose)                                                       
                                                                 
 batch_normalization_9 (Batc  (None, 90, 60, 128)      512       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_13 (LeakyReLU)  (None, 90, 60, 128)       0         
                                                                 
 conv2d_transpose_7 (Conv2DT  (None, 90, 60, 128)      409600    
 ranspose)                                                       
                                                                 
 batch_normalization_10 (Bat  (None, 90, 60, 128)      512       
 chNormalization)                                                
                                                                 
 leaky_re_lu_14 (LeakyReLU)  (None, 90, 60, 128)       0         
                                                                 
 conv2d_transpose_8 (Conv2DT  (None, 180, 120, 64)     204800    
 ranspose)                                                       
                                                                 
 batch_normalization_11 (Bat  (None, 180, 120, 64)     256       
 chNormalization)                                                
                                                                 
 leaky_re_lu_15 (LeakyReLU)  (None, 180, 120, 64)      0         
                                                                 
 conv2d_transpose_9 (Conv2DT  (None, 360, 240, 3)      4800      
 ranspose)                                                       
                                                                 
=================================================================
Total params: 37,639,104
Trainable params: 37,637,952
Non-trainable params: 1,152
_________________________________________________________________                                                                               
```
* # Discriminator
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 180, 120, 64)      4864      
                                                                 
 leaky_re_lu_16 (LeakyReLU)  (None, 180, 120, 64)      0         
                                                                 
 batch_normalization_12 (Bat  (None, 180, 120, 64)     256       
 chNormalization)                                                
                                                                 
 conv2d_5 (Conv2D)           (None, 90, 60, 128)       204928    
                                                                 
 leaky_re_lu_17 (LeakyReLU)  (None, 90, 60, 128)       0         
                                                                 
 batch_normalization_13 (Bat  (None, 90, 60, 128)      512       
 chNormalization)                                                
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 18, 12, 128)      0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 18, 12, 128)       0         
                                                                 
 conv2d_6 (Conv2D)           (None, 18, 12, 128)       409728    
                                                                 
 leaky_re_lu_18 (LeakyReLU)  (None, 18, 12, 128)       0         
                                                                 
 batch_normalization_14 (Bat  (None, 18, 12, 128)      512       
 chNormalization)                                                
                                                                 
 conv2d_7 (Conv2D)           (None, 9, 6, 256)         819456    
                                                                 
 leaky_re_lu_19 (LeakyReLU)  (None, 9, 6, 256)         0         
                                                                 
 batch_normalization_15 (Bat  (None, 9, 6, 256)        1024      
 chNormalization)                                                
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 1, 1, 256)        0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 1, 1, 256)         0         
                                                                 
 flatten_1 (Flatten)         (None, 256)               0         
                                                                 
 dense_5 (Dense)             (None, 256)               65792     
                                                                 
 leaky_re_lu_20 (LeakyReLU)  (None, 256)               0         
                                                                 
 dense_6 (Dense)             (None, 256)               65792     
                                                                 
 leaky_re_lu_21 (LeakyReLU)  (None, 256)               0         
                                                                 
 dense_7 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 1,573,121
Trainable params: 1,571,969
Non-trainable params: 1,152
_________________________________________________________________

```

