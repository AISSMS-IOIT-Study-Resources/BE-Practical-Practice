import tensorflow as tf
import numpy as np

def tensor_creation():
    print("=== Tensor Creation ===")
    scaler = tf.constant(5)
    vector = tf.constant([1, 2, 3])
    matrix = tf.constant([[1, 2], [3, 4]])
    tensor_3d = tf.constant([[[1], [2]], [[3], [4]]])

    print(f"Scaler: {scaler}")
    print(f"Vector: {vector}")
    print(f"Matrix: {matrix}")
    print(f"Tensor-3D: {tensor_3d}")

    np_array = np.array([[10, 20], [30, 40]])
    tensor_from_np = tf.convert_to_tensor(np_array)
    print(f"Tensor from NumPy: {tensor_from_np}")

def tensor_manipulation():
    print("\n=== Tensor Manipulation ===")
    a = tf.ones((2, 3))
    b = tf.zeros((2, 3))
    c = tf.fill((2, 3), 7)

    print(f"Ones: {a}")
    print(f"Zeros: {b}")
    print(f"Fills: {c}")

    reshaped = tf.reshape(c, (3, 2))
    print(f"Reshaped to (3, 2):\n{reshaped}")

    print(f"First row of a: {a[0]}")
    print(f"Element at (1, 2): {a[1, 2].numpy()}")

    concat = tf.concat([a, b], axis=0)
    print(f"Concatenated (along row):\n{concat}")

def math_operations():
    print("\n=== Math Operations ===")
    x = tf.constant([2.0, 4.0, 6.0])
    y = tf.constant([1.0, 3.0, 5.0])

    print(f"x + y = {tf.add(x, y)}")
    print(f"x - y = {tf.subtract(x, y)}")
    print(f"x * y = {tf.multiply(x, y)}")
    print(f"x / y = {tf.divide(x, y)}")
    print(f"Dot product = {tf.tensordot(x, y, axes=1)}")

    mat = tf.constant([[1], [2], [3]])
    vec = tf.constant([4, 5, 6])
    print(f"Broadcasted addition:\n{mat + vec}")

@tf.function
def compute_graph(a, b):
    return tf.sqrt(tf.add(a ** 2, b ** 2)) # Hypotenuse

def dynamic_sum(n):
    total = tf.constant(0)
    for i in range(n):
        total += 1
        tf.print("Step", i, "Total: ", total)
    return total

def main():
    tensor_creation()
    tensor_manipulation()
    math_operations()

    print("\n=== Computation Graph ===")
    result = compute_graph(3.0, 4.0)
    print(f"Hypotenuse (graph mode): {result}")

    print("=== Eager Execution Example ===")
    final_sum = dynamic_sum(5)  
    print(f"Final sum: {final_sum.numpy()}")

main()