import numpy as np
from scipy.sparse import csr_matrix


dense_vector = np.array([0, 0, 1, 0, 2, 0, 3])

sparse_vector = csr_matrix(dense_vector)

dense_again = sparse_vector.toarray()

sparse_vector2 = csr_matrix(np.array([0, 1, 0, 0, 1, 0, 1]))
sum_vector = sparse_vector + sparse_vector2

print(sum_vector.toarray())

dot_product = sparse_vector.dot(sparse_vector2.T)
print(dot_product.toarray())