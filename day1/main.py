import numpy as np

apple = np.array([6, 7])
orange = np.array([7, 8])
lemon = np.array([1, 4])

print(f"Apple vector:  {apple}")
print(f"Orange vector: {orange}")
print(f"Lemon vector:  {lemon}")

print(f"\nType of apple: {type(apple)}")

print(f"Shape of apple: {apple.shape}")
# Output: (2,)  — this means it has 2 elements (a 2D vector)

big_vector = np.random.rand(384)  # 384 random numbers between 0 and 1
print(f"\n384-dim vector (first 10 values): {big_vector[:10]}")
print(f"Shape: {big_vector.shape}")
# Output: (384,) — 384 elements, which is what our embedding model produces


#   [6, 7] + [7, 8] = [6+7, 7+8] = [13, 15]
# The DIFFERENCE between two vectors tells us how far apart they are. If apple - orange = small numbers, they're close (similar).

vec_a = np.array([3.0, 4.0, 5.0])
vec_b = np.array([1.0, 2.0, 3.0])

print(f"a = {vec_a}")
print(f"b = {vec_b}")
print(f"a + b = {vec_a + vec_b}")  # [4.0, 6.0, 8.0]

# Subtraction: finds the difference element by element
print(f"a - b = {vec_a - vec_b}")  # [2.0, 2.0, 2.0]

# Multiplication by a number (scalar multiplication)
# This scales the vector — makes it longer or shorter
print(f"a * 2 = {vec_a * 2}")  # [6.0, 8.0, 10.0]

# Element-wise multiplication (NOT the dot product — we'll cover that next)
print(f"a * b = {vec_a * vec_b}")  # [3.0, 8.0, 15.0]


# ─────────────────────────────────────────────────────────────
# STEP 4: The Dot Product
# ─────────────────────────────────────────────────────────────
# The dot product is THE most important operation in vector math.
# It's used inside every single similarity calculation.
#
# How it works:
#   1. Multiply corresponding elements together
#   2. Add up all the results
#
# Example: [3, 4, 5] · [1, 2, 3]
#   Step 1: 3×1=3, 4×2=8, 5×3=15
#   Step 2: 3 + 8 + 15 = 26
#
# The dot product tells us something about how much two vectors
# "agree" — how much they point in the same direction.
# Higher dot product = more similar direction.

# Let's calculate it manually first
manual_dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]
print(
    f"Manual dot product: {vec_a[0]}×{vec_b[0]} + {vec_a[1]}×{vec_b[1]} + {vec_a[2]}×{vec_b[2]} = {manual_dot}"
)

# Now the NumPy way (one line, much faster for big vectors)
numpy_dot = np.dot(vec_a, vec_b)
print(f"NumPy dot product:  {numpy_dot}")

# They're the same! NumPy just does it faster, especially for vectors with hundreds of dimensions.

# Let's see how the dot product behaves with our fruit vectors
print(f"\nApple · Orange = {np.dot(apple, orange)}")
print(f"Apple · Lemon  = {np.dot(apple, lemon)}")
# Apple·Orange should be bigger because they're more similar!


print("STEP 4: Vector Magnitude (Norm) — How 'Long' is a Vector?")

# ─────────────────────────────────────────────────────────────
# STEP 5: Vector Magnitude (also called Norm or Length)
# ─────────────────────────────────────────────────────────────
# The magnitude of a vector is its "length" — how far it is
# from the origin (the zero point).
#
# For a 2D vector [a, b], magnitude = √(a² + b²)
# This is just the Pythagorean theorem!
#
# For a 3D vector [a, b, c], magnitude = √(a² + b² + c²)
# Same pattern, just more terms.
#
# General formula: magnitude = √(sum of all elements squared)
#
# Why does this matter?
# We need magnitudes to calculate cosine similarity. Magnitude is used to "normalize" vectors — make them all the same length so we can fairly compare their directions.
# Manual calculation for vec_a = [3, 4, 5]

import math

manual_magnitude = math.sqrt(3**2 + 4**2 + 5**2)
print(
    f"Manual magnitude of {vec_a}: √(3² + 4² + 5²) = √{3**2 + 4**2 + 5**2} = {manual_magnitude:.4f}"
)

# NumPy way — np.linalg.norm() calculates the magnitude
# "linalg" stands for "linear algebra"
# "norm" is the mathematical term for magnitude/length
numpy_magnitude = np.linalg.norm(vec_a)
print(f"NumPy magnitude:    {numpy_magnitude:.4f}")

# Let's find magnitudes of our fruit vectors too
print(f"\nMagnitude of Apple  {apple}: {np.linalg.norm(apple):.4f}")
print(f"Magnitude of Orange {orange}: {np.linalg.norm(orange):.4f}")
print(f"Magnitude of Lemon  {lemon}: {np.linalg.norm(lemon):.4f}")


print("STEP 5: Unit Vectors — Normalising to Length 1")

# ─────────────────────────────────────────────────────────────
# STEP 6: Normalisation (Creating Unit Vectors)
# ─────────────────────────────────────────────────────────────
# A "unit vector" is a vector with magnitude = 1.
# We create one by dividing each element by the magnitude.
#
# Why normalise?
# Imagine two documents about cooking. One is a 10-page essay,
# the other is a 2-sentence recipe. The essay's vector will be
# "longer" simply because it has more words, not because it's
# more relevant. Normalising removes this size bias so we
# compare MEANING (direction) rather than LENGTH (amount).

# Normalising vec_a: divide each element by the magnitude

unit_a = vec_a / np.linalg.norm(vec_a)
print(f"Original vec_a:    {vec_a}")
print(f"Magnitude:         {np.linalg.norm(vec_a):.4f}")
print(f"Unit vector:       {unit_a}")
print(f"New magnitude:     {np.linalg.norm(unit_a):.4f}")  # Should be 1.0!

# Let's normalise our fruit vectors
apple_unit = apple / np.linalg.norm(apple)
orange_unit = orange / np.linalg.norm(orange)
lemon_unit = lemon / np.linalg.norm(lemon)

print(f"\nApple  (normalised): {apple_unit}")
print(f"Orange (normalised): {orange_unit}")
print(f"Lemon  (normalised): {lemon_unit}")

print("STEP 6: Putting It All Together — Why This Matters")

# ─────────────────────────────────────────────────────────────
# STEP 7: Preview of Cosine Similarity
# ─────────────────────────────────────────────────────────────
# Cosine similarity uses BOTH dot product AND magnitude.
# Formula: cosine_similarity(A, B) = (A · B) / (|A| × |B|)
#
# Where:
#   A · B    = dot product of A and B
#   |A|      = magnitude of A
#   |B|      = magnitude of B
#
# This gives a value between -1 and 1:
#   1  = perfectly similar (vectors point same direction)
#   0  = no similarity (vectors are perpendicular)
#  -1  = opposite meaning (vectors point opposite directions)
#
# Let's try it with our fruits!


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    This is the core function of any vector database!

    Parameters:
        vec1: First vector (NumPy array)
        vec2: Second vector (NumPy array)

    Returns:
        A float between -1 and 1 (closer to 1 = more similar)
    """
    dot_product = np.dot(vec1, vec2)  # Numerator
    magnitude1 = np.linalg.norm(vec1)  # Part of denominator
    magnitude2 = np.linalg.norm(vec2)  # Part of denominator

    # Avoid division by zero (if a vector is all zeros)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# Now let's measure similarity between our fruits
sim_apple_orange = cosine_similarity(apple, orange)
sim_apple_lemon = cosine_similarity(apple, lemon)
sim_orange_lemon = cosine_similarity(orange, lemon)

print("Similarity Scores (1.0 = identical, 0.0 = unrelated):")
print(
    f"  Apple  ↔ Orange: {sim_apple_orange:.4f}  (both sweet & round — very similar!)"
)
print(f"  Apple  ↔ Lemon:  {sim_apple_lemon:.4f}  (somewhat similar)")
print(f"  Orange ↔ Lemon:  {sim_orange_lemon:.4f}  (least similar of the three)")

print("\nSTEP 7: Simulating a Mini Vector Search")

# ─────────────────────────────────────────────────────────────
# STEP 8: Your First Vector Search!
# ─────────────────────────────────────────────────────────────
# This is what a vector db does at its core:
#   1. Store a bunch of vectors
#   2. When given a query vector, find the most similar ones
#
# Let's simulate this with our fruit vectors.

# Our "database" — a dictionary mapping names to vectors
fruit_database = {
    "apple": np.array([6, 7]),
    "orange": np.array([7, 8]),
    "lemon": np.array([1, 4]),
    "banana": np.array([5, 2]),
    "watermelon": np.array([9, 9]),
    "grapefruit": np.array([2, 5]),
    "mango": np.array([8, 6]),
    "kiwi": np.array([4, 5]),
}

# The "query" — we want to find fruits most similar to this
query = np.array([7, 7])  # Something sweet and round
print(f"Query vector: {query}  (a sweet, round fruit)")
print(f"\nSearching database of {len(fruit_database)} fruits...\n")

# Calculate similarity between query and every fruit in the database
results = []
for name, vector in fruit_database.items():
    similarity = cosine_similarity(query, vector)
    results.append((name, similarity, vector))

# Sort by similarity (highest first) — this is the "ranking" step
results.sort(key=lambda x: x[1], reverse=True)

# Display the ranked results (like search results!)
print(f"{'Rank':<6} {'Fruit':<14} {'Vector':<16} {'Similarity':<12}")
print("-" * 48)
for i, (name, sim, vec) in enumerate(results, 1):
    bar = "█" * int(sim * 20)  # Visual similarity bar
    print(f"  {i:<4} {name:<14} {str(vec):<16} {sim:.4f}  {bar}")

print("\n✓ The most similar fruit to [7,7] is:", results[0][0])
print("  This is EXACTLY how a vector db search works!")
