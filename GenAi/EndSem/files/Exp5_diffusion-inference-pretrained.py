import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return tf.linspace(beta_start, beta_end, timesteps)

# DDPM parameters
TIMESTEPS = 1000
betas = linear_beta_schedule(TIMESTEPS)
alphas = 1 - betas
alphas_cumprod = tf.math.cumprod(alphas)

def get_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(28*28, activation='linear')(x)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(x)
    return tf.keras.Model(inputs, outputs)

model = get_model()
def sample(model, shape=(28, 28, 1), timesteps=TIMESTEPS):
    x = tf.random.normal((1, *shape)) # start from random noise

    for t in reversed(range(timesteps)):
        predicted_noise = model(x, training=False)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        coef1 = 1 / tf.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t)

        x = coef1 * (x - coef2 * predicted_noise)

        if t > 0:
            noise = tf.random.normal(shape=(1, *shape))
            sigma_t = tf.sqrt(beta_t)
            x += sigma_t * noise

    return x

def main():
    generated_image = sample(model)
    generated_image = (generated_image + 1) / 2

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.title("Generated MNIST-like Image (DDPM)")
    plt.axis('off')
    plt.show()

main()


# # pip install tensorflow numpy
# import tensorflow as tf
# import numpy as np

# # Simple denoiser model (small conv autoencoder)
# def get_denoiser(input_shape=(28,28,1)):
#     inputs = tf.keras.Input(shape=input_shape)
#     x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
#     x = tf.keras.layers.MaxPool2D()(x)
#     x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#     x = tf.keras.layers.UpSampling2D()(x)
#     x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
#     x = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
#     return tf.keras.Model(inputs, x)

# # Simple inference demo: add noise to an MNIST sample and denoise
# def demo_inference():
#     (x_train, _), _ = tf.keras.datasets.mnist.load_data()
#     x = x_train[:16].astype('float32') / 255.0
#     x = np.expand_dims(x, -1)

#     denoiser = get_denoiser()
#     # For demo purposes, we'll train for a very small number of steps so inference runs
#     denoiser.compile(optimizer='adam', loss='mse')

#     # Create noisy targets and train briefly (illustrative only)
#     noisy = x + np.random.normal(0, 0.5, size=x.shape)
#     noisy = np.clip(noisy, 0.0, 1.0)

#     denoiser.fit(noisy, x, epochs=1, batch_size=8)

#     # Inference: denoise a random noisy image
#     sample = noisy[0:1]
#     denoised = denoiser.predict(sample)
#     print('Noisy sample min/max:', sample.min(), sample.max())
#     print('Denoised sample min/max:', denoised.min(), denoised.max())

# if __name__ == '__main__':
#     demo_inference()