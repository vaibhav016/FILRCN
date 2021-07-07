import tensorflow as tf
import numpy as np

def norm_filter_direction(copy_of_the_weights):
    direction1 = []
    direction2 = []

    for w in copy_of_the_weights:
        if len(w.shape) == 3:
            random_vector1 = tf.random.normal(w.shape, 0, 1, tf.float32)
            random_vector2 = tf.random.normal(w.shape, 0, 1, tf.float32)

            normalised_current = tf.norm(w)
            normalised_random_vec1 = tf.norm(random_vector1) + 1e-8
            normalised_random_vec2 = tf.norm(random_vector2) + 1e-8

            random_vector1 = random_vector1 * (normalised_current / normalised_random_vec1)
            random_vector2 = random_vector2 * (normalised_current / normalised_random_vec2)

            direction1.append(random_vector1)
            direction2.append(random_vector2)

        else:
            direction1.append(tf.zeros_like(w))
            direction2.append(tf.zeros_like(w))

    return direction1, direction2


def pca_filter_direction(copy_of_the_weights):
    random_direction1 = []
    random_direction2 = []

    for w in copy_of_the_weights:
        if not len(w.shape) == 3:
            random_direction1.append(tf.zeros_like(w))
            random_direction1.append(tf.zeros_like(w))

        else:
            random_vector = tf.identity(w)
            random_vector1 = random_vector - tf.math.reduce_mean(random_vector, (1, 2), keepdims=True)
            random_vector2 = random_vector - tf.math.reduce_mean(random_vector, (0, 1), keepdims=True)

            random_vector2 = np.transpose(random_vector2, (2, 0, 1))

            s1, u1, v1 = tf.linalg.svd(random_vector1, False)

            s2, u2, v2 = tf.linalg.svd(random_vector2, False)


            random_vector1 = u1 @ tf.linalg.diag(s1)[:, :, :1] @ tf.transpose(v1, (0, 2, 1))[:, :1, :]
            random_vector2 = u2 @ tf.linalg.diag(s2)[:, :, :1] @ tf.transpose(v2, (0, 2, 1))[:, :1, :]
            random_vector2 = tf.transpose(random_vector2, (1, 2, 0))


            w_norm = tf.norm(w)

            d_norm1 = tf.norm(random_vector1)
            d_norm2 = tf.norm(random_vector2)

            random_vector1 = random_vector1 * (w_norm / (d_norm1 + 1e-10))
            random_vector2 = random_vector2 * (w_norm / (d_norm2 + 1e-10))

            random_direction1.append(random_vector1)
            random_direction2.append(random_vector2)

    return random_direction1, random_direction2

