import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CubicDenseThingy(tf.keras.layers.Layer):
    def __init__(self, out = 1, trainable=True, name="CubicThingy", dtype=tf.float32, dynamic=False, **kwargs):
        self.out = out
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight("bias", [], trainable=True, initializer="random_normal")
        self.kernel_linear = self.add_weight("linear_kernel", [], trainable=True, initializer="random_normal")
        self.kernel_quad = self.add_weight("quad_kernel", [], trainable=True, initializer="random_normal")
        self.kernel_cubic = self.add_weight("cubic_kernel", [], trainable=True, initializer="random_normal")

    def call(self, inputs):
        return (inputs ** 3) * self.kernel_cubic + (inputs ** 2) * self.kernel_quad + (inputs) * self.kernel_linear + self.bias

model = tf.keras.Sequential(
    CubicDenseThingy(input_shape=[])
)

# print(model)
# print(model.layers)
print(model.trainable_variables)

Mse = tf.keras.losses.MeanSquaredError()

def loss(model, inp, t):
    cbl = model.layers[0]
    with tf.GradientTape() as tape:
        l = tf.reduce_mean(tf.losses.mean_squared_logarithmic_error(inp, model(inp)))
        l = l + (tf.reduce_sum(tf.abs(cbl.kernel_quad)) + tf.reduce_sum(tf.abs(cbl.kernel_cubic) * 2))
    return l, tape.gradient(l, model.trainable_variables)

q = 0.001
c = 0.0001
slope = 0.269
yinter = 1.35
n = 100
error = np.random.normal(scale=1, size=n)

x_val = np.arange(n)
y_val = x_val * (x_val * (x_val * c + q) + slope) + yinter

y_val_error = y_val + error

# arr = np.stack((x_val, y_val_error))

# plt.scatter(x_val, y_val_error)
# plt.show()

lt = []
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
for i in range(400):
    l, g = loss(model, x_val, y_val_error)
    opt.apply_gradients(zip(g, model.trainable_variables))
    lt.append(l)
    print(f"Epoch: {i} | Loss: {l}")
    print("=" * 20)

f, a = plt.subplots(1, 2)
a[0].scatter(x_val, y_val_error)
a[0].plot((x:=tf.linspace(0.0, 100, 100)), model.predict(x))

a[1].plot(lt)
plt.show()

print(model.trainable_variables)
