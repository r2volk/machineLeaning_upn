import time, psutil, numpy as np, tensorflow as tf
from sklearn.datasets import make_classification

print(f"CPU cores: {psutil.cpu_count()}, RAM total: {psutil.virtual_memory().total/1e9:.1f} GB")
print(f"TF {tf.__version__} devices: {tf.config.list_physical_devices('CPU')}")

# 1. NumPy matmul (CPU)
for n in [1000, 2000, 4000]:
    a = np.random.randn(n,n).astype(np.float32)
    t0=time.time(); c=a@a
    print(f"NumPy {n}x{n} matmul: {time.time()-t0:.2f}s RAM {psutil.virtual_memory().percent}%")

# 2. TensorFlow matmul (CPU)
for n in [1000, 2000, 4000]:
    a=tf.random.normal((n,n)); t0=time.time(); c=tf.matmul(a,a); c.numpy()
    print(f"TF {n}x{n} matmul: {time.time()-t0:.2f}s RAM {psutil.virtual_memory().percent}%")

# 3. Entrenamiento escalonado (mide hasta dónde aguanta)
for samples in [10000, 50000, 100000]:
    X,y = make_classification(n_samples=samples, n_features=20, n_informative=15, random_state=0)
    X=X.astype(np.float32)
    model=tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu',input_shape=(20,)),
                               tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer='adam',loss='binary_crossentropy')
    t0=time.time(); model.fit(X,y,epochs=3, batch_size=1024, verbose=0)
    print(f"Train {samples} samples: {time.time()-t0:.2f}s RAM {psutil.virtual_memory().percent}%")
    if psutil.virtual_memory().percent > 85:
        print("RAM >85% - paro para no colgar"); break