from pathlib import Path

import numpy as np

print("=" * 60)
print("ESSENTIAL NUMPY FOR VECTOR DBs")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. CREATING ARRAYS — Different Ways
# ─────────────────────────────────────────────────────────────
print("\n--- 1. Creating Arrays ---")

# From a Python list (most common)
v1 = np.array([1.0, 2.0, 3.0])
print(f"From list:    {v1}")

# Array of zeros (useful for initialising empty vectors)
zeros = np.zeros(5)
print(f"Zeros:        {zeros}")

# Array of ones
ones = np.ones(5)
print(f"Ones:         {ones}")

# Random values between 0 and 1 (useful for testing)
rand = np.random.rand(5)
print(f"Random:       {rand}")

# A specific data type — float32 uses less memory than float64
# This matters when storing millions of vectors!
v_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print(f"Float32:      {v_float32}  (dtype: {v_float32.dtype})")

v_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(f"Float64:      {v_float64}  (dtype: {v_float64.dtype})")


# ─────────────────────────────────────────────────────────────
# 2. ARRAY SHAPES — Understanding Dimensions
# ─────────────────────────────────────────────────────────────
print("\n--- 2. Array Shapes ---")

# A single vector (1D array)
single = np.array([1, 2, 3, 4, 5])
print(f"1D array: shape={single.shape}, ndim={single.ndim}")
# shape=(5,) means 5 elements in one dimension

# A matrix of vectors (2D array) — like storing multiple vectors
# This is how our database will store all vectors together!
# Each ROW is one vector.
matrix = np.array(
    [
        [1.0, 2.0, 3.0],  # vector 0
        [4.0, 5.0, 6.0],  # vector 1
        [7.0, 8.0, 9.0],  # vector 2
    ]
)
print(f"2D array: shape={matrix.shape}, ndim={matrix.ndim}")
# shape=(3, 3) means 3 rows (vectors) × 3 columns (dimensions)

# Accessing specific vectors from the matrix
print(f"  Row 0 (first vector):  {matrix[0]}")
print(f"  Row 1 (second vector): {matrix[1]}")
print(f"  Row 2 (third vector):  {matrix[2]}")


# ─────────────────────────────────────────────────────────────
# 3. STACKING VECTORS — Building Up Your Database
# ─────────────────────────────────────────────────────────────
print("\n--- 3. Stacking Vectors ---")

# When users insert vectors one at a time, we need to combine them.
# np.vstack "vertically stacks" arrays into a matrix.
vec_a = np.array([1.0, 2.0, 3.0])
vec_b = np.array([4.0, 5.0, 6.0])
vec_c = np.array([7.0, 8.0, 9.0])

# Stack them into a matrix (each vector becomes a row)
stacked = np.vstack([vec_a, vec_b, vec_c])
print(f"Stacked shape: {stacked.shape}")  # (3, 3)
print(f"Stacked matrix:\n{stacked}")

# Adding one more vector to an existing matrix
vec_d = np.array([10.0, 11.0, 12.0])
stacked = np.vstack([stacked, vec_d.reshape(1, -1)])
# reshape(1, -1) turns [10, 11, 12] into [[10, 11, 12]]
# This is needed because vstack needs matching dimensions
print(f"\nAfter adding vec_d: shape={stacked.shape}")  # (4, 3)
print(f"{stacked}")


# ─────────────────────────────────────────────────────────────
# 4. BATCH DOT PRODUCTS — Search All at Once
# ─────────────────────────────────────────────────────────────
print("\n--- 4. Batch Dot Products")

# In a database with 10,000 vectors, computing similarity one
# by one in a Python loop would be painfully slow.
# NumPy can compute ALL dot products at once using matrix math!

# Our "database" of vectors
database = np.array(
    [
        [1.0, 0.0, 0.0],  # vector 0: points along x-axis
        [0.0, 1.0, 0.0],  # vector 1: points along y-axis
        [1.0, 1.0, 0.0],  # vector 2: between x and y
        [0.5, 0.5, 0.5],  # vector 3: diagonal
    ]
)

# Our query vector
query = np.array([1.0, 1.0, 0.0])

# SLOW WAY: Loop through each vector (don't do this!)
print("Slow way (loop):")
for i, vec in enumerate(database):
    dot = np.dot(query, vec)
    print(f"  query · vector_{i} = {dot:.2f}")

# FAST WAY: Matrix multiplication does ALL dot products at once
# database @ query  computes dot product of query with every row
print("\nFast way (matrix multiplication):")
all_dots = database @ query
print(f"  All dot products at once: {all_dots}")
# Result: [1.0, 1.0, 2.0, 1.0]
# This is EXACTLY the same as the loop, but 100x faster for big data!

# Even faster: using np.dot with 2D arrays
all_dots_v2 = np.dot(database, query)
print(f"  Using np.dot:            {all_dots_v2}")


# ─────────────────────────────────────────────────────────────
# 5. BATCH NORMS — Normalise All Vectors at Once
# ─────────────────────────────────────────────────────────────
print("\n--- 5. Batch Norms ---")
print("db ", database)
# Calculate magnitude of every vector in the database at once
# axis=1 means "compute along each row" (each row is a vector)
all_norms = np.linalg.norm(database, axis=1)
print(f"All magnitudes: {all_norms}")

# Normalise all vectors at once (make them all length 1)
# We reshape norms to be (4,1) so division broadcasts correctly
# "Broadcasting" means NumPy auto-expands dimensions to match
normalised_db = database / all_norms[:, np.newaxis]
print(f"\nNormalised database:\n{normalised_db}")

# Verify they're all length 1 now
new_norms = np.linalg.norm(normalised_db, axis=1)
print(f"New magnitudes: {new_norms}")  # All should be 1.0

# ─────────────────────────────────────────────────────────────
# 5.1 TRUE COSINE SIMILARITY — Normalise the Query Too
# ─────────────────────────────────────────────────────────────
print("\n--- 5.1 True Cosine Similarity ---")

# Cosine similarity:
# cos(q, v) = (q · v) / (||q|| * ||v||)
# If BOTH vectors are unit length, then cosine similarity = dot product.
query_norm = np.linalg.norm(query)
normalised_query = query / query_norm
print(f"Query magnitude:           {query_norm:.4f}")
print(f"Normalised query:          {normalised_query}")

# Verify the query is length 1 now
normalised_query_norm = np.linalg.norm(normalised_query)
print(f"Normalised query norm:     {normalised_query_norm:.4f}")

# If only the database is normalised, scores are still scaled by ||query||
db_only_normalised_scores = normalised_db @ query

# True cosine similarity: normalise BOTH database and query
cosine_scores = normalised_db @ normalised_query

print(f"\nDB-only normalised scores: {db_only_normalised_scores}")
print(f"True cosine similarities:  {cosine_scores}")

print("\nCompare the two:")
for i, (db_score, cos_score) in enumerate(
    zip(db_only_normalised_scores, cosine_scores)
):
    print(f"  vector_{i}: db-only={db_score:.4f}   cosine={cos_score:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. ARGSORT — Finding Top-K Results
# ─────────────────────────────────────────────────────────────
print("\n--- 6. Finding Top-K Results ---")

# After computing all similarities, we need the TOP K results.
# np.argsort returns the INDICES that would sort the array.

scores = np.array([0.3, 0.9, 0.1, 0.7, 0.5])
names = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]

# argsort gives indices from lowest to highest
sorted_indices = np.argsort(scores)
print(f"Scores:      {scores}")
print(f"Sorted idx:  {sorted_indices}")
# [2, 0, 4, 3, 1] means index 2 is smallest, index 1 is largest

# For similarity, we want HIGHEST first, so we reverse it
top_indices = np.argsort(scores)[::-1]
print(f"Top to bottom: {top_indices}")

# Get top 3 results
k = 3
top_k_indices = np.argsort(scores)[::-1][:k]
# print top-k results
print(f"\nTop {k} results:")
for rank, idx in enumerate(top_k_indices, 1):
    print(f"  #{rank}: {names[idx]} (score: {scores[idx]:.2f})")


# ─────────────────────────────────────────────────────────────
# 7. SAVING AND LOADING — Persistence
# ─────────────────────────────────────────────────────────────
print("\n--- 7. Saving and Loading Vectors ---")

# When we build our database, vectors need to survive restarts.
# NumPy can save/load arrays to binary files extremely fast.
# Save them inside this project so you can actually see the files.
base_dir = Path(__file__).resolve().parent
vectors_path = base_dir / "test_vectors.npy"
db_bundle_path = base_dir / "test_db.npz"

# Save a single array
np.save(vectors_path, database)
print(f"Saved database to {vectors_path}")

# Load it back
loaded = np.load(vectors_path)
print(f"Loaded shape: {loaded.shape}")
print(f"Arrays match: {np.array_equal(database, loaded)}")

# Save multiple arrays at once (useful for saving vectors + norms +
# cosine-ready data)
np.savez(
    db_bundle_path,
    vectors=database,
    norms=all_norms,
    normalised_vectors=normalised_db,
    query=query,
    normalised_query=normalised_query,
    cosine_scores=cosine_scores,
)
print(f"\nSaved multiple arrays to {db_bundle_path}")

# Load multiple arrays
data = np.load(db_bundle_path)
print(f"Loaded keys: {list(data.keys())}")
print(f"Vectors shape:            {data['vectors'].shape}")
print(f"Norms shape:              {data['norms'].shape}")
print(f"Normalised vectors shape: {data['normalised_vectors'].shape}")
print(f"Query shape:              {data['query'].shape}")
print(f"Normalised query shape:   {data['normalised_query'].shape}")
print(f"Cosine scores shape:      {data['cosine_scores'].shape}")
