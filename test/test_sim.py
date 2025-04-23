import os
import sys

# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import scipy.sparse as sp
import data_utils  # From parent directory

def test_u3_similarity():
    """
    Test function to verify the similarity vector computation for u_3
    exactly matching the data in the screenshot.
    """
    print("Testing u_3 similarity computation...")
    
    # Create interaction matrix exactly as shown in screenshot
    # The data shows columns 1-8 and users u_0 through u_7
    # where 1s and 0s indicate interactions
    interaction_data = np.array([
        # Columns 1  2  3  4  5  6  7  8  (indexed as 0-7 in code)
        [1, 0, 1, 0, 0, 0, 1, 0],  # u_0
        [1, 1, 1, 0, 0, 1, 0, 1],  # u_1
        [0, 0, 0, 1, 1, 0, 0, 0],  # u_2
        [0, 0, 1, 1, 0, 1, 0, 1],  # u_3 (CORRECTED: 00110101)
        [1, 1, 1, 0, 0, 1, 1, 0],  # u_4
        [0, 0, 0, 0, 0, 1, 1, 1],  # u_5
        [1, 0, 0, 0, 0, 0, 1, 1],  # u_6
        [0, 1, 1, 1, 0, 1, 1, 0]   # u_7
    ])
    
    # Convert to sparse matrix as expected by the functions
    train_data = sp.csr_matrix(interaction_data)
    
    # Verify the interaction matrix
    print("\nUser-Item Interaction Matrix:")
    print(train_data.toarray())
    print("\nSpecifically, u_3's interactions:", interaction_data[3])
    
    # Compute similarity matrix
    similarity_matrix = data_utils.compute_similarity_matrix(train_data, method='cosine')
    
    # Print the similarity matrix focusing on u_3
    print("\nUser Similarity Matrix (Cosine values rounded to 3 decimals):")
    print(np.round(similarity_matrix, 3))
    
    # Validate similarity scores specifically for u_3
    expected_similarities = {
        (3, 0): 0.289,    # u_3 and u_0 - not directly from screenshot
        (3, 1): 0.671,  # u_3 and u_1
        (3, 2): 0.354,  # u_3 and u_2
        (3, 4): 0.447,  # u_3 and u_4
        (3, 5): 0.577,  # u_3 and u_5
        (3, 6): 0.289,  # u_3 and u_6
        (3, 7): 0.671   # u_3 and u_7
    }
    
    print("\nValidating u_3 similarity scores:")
    for (u1, u2), expected in expected_similarities.items():
        actual = similarity_matrix[u1, u2]
        print(f"Similarity between u_{u1} and u_{u2}: {actual:.3f} (Expected: {expected:.3f})")
        assert abs(actual - expected) < 0.01, f"Similarity between u_{u1} and u_{u2} doesn't match expected value"
    
    # Test the top-3 similar users for u_3
    print("\nTesting top-3 similar users for u_3:")
    top_indices, top_similarities = data_utils.get_top_k_similar_users(similarity_matrix, 3, k=3)
    print(f"Top 3 similar users for u_3: {['u_' + str(idx) for idx in top_indices]}")
    print(f"Similarity scores: {[round(s, 3) for s in top_similarities]}")
    
    # Expected top users from screenshot: u_1, u_7, u_5
    expected_top_users = [1, 7, 5]
    for idx, user_id in enumerate(expected_top_users):
        assert user_id in top_indices, f"Expected user u_{user_id} to be in top-3 for u_3"
    
    # Test weighted similarity vector computation
    print("\nTesting weighted similarity vector computation for u_3:")
    weighted_vector = data_utils.create_weighted_similarity_vector(similarity_matrix, 3, top_k=3, train_data=train_data)
    print(f"Weighted similarity vector for u_3: {np.round(weighted_vector, 3)}")
    
    # Expected vector from screenshot (Final vector row)
    expected_vector = np.array([0.671, 1.342, 1.342, 0.671, 0.000, 1.919, 1.248, 1.248])
    
    print("\nComponent-wise breakdown of the weighted vector:")
    # Vector component for u_1 (weight 0.671)
    u1_vector = train_data[1].toarray()[0] * 0.671
    print(f"u_1 contribution (weight 0.671): {np.round(u1_vector, 3)}")
    
    # Vector component for u_7 (weight 0.671)
    u7_vector = train_data[7].toarray()[0] * 0.671
    print(f"u_7 contribution (weight 0.671): {np.round(u7_vector, 3)}")
    
    # Vector component for u_5 (weight 0.577)
    u5_vector = train_data[5].toarray()[0] * 0.577
    print(f"u_5 contribution (weight 0.577): {np.round(u5_vector, 3)}")
    
    # Compare with the expected sum
    expected_sum = u1_vector + u7_vector + u5_vector
    print(f"\nExpected sum: {np.round(expected_sum, 3)}")
    print(f"Actual weighted vector: {np.round(weighted_vector, 3)}")
    
    # Compare the actual vector with expected vector from screenshot
    print("\nComparison with expected vector from screenshot:")
    print(f"Expected: {expected_vector}")
    print(f"Actual:   {np.round(weighted_vector, 3)}")
    
    # Check if vectors match (with small tolerance for rounding)
    vector_match = np.allclose(weighted_vector, expected_vector, atol=0.01)
    if vector_match:
        print("\n✅ Success! Weighted similarity vector matches the expected values from the screenshot.")
    else:
        print("\n❌ Error: Weighted similarity vector doesn't match expected values.")
        # Show specific differences
        for i in range(len(expected_vector)):
            if abs(weighted_vector[i] - expected_vector[i]) >= 0.01:
                print(f"  Position {i}: Expected {expected_vector[i]:.3f}, Got {weighted_vector[i]:.3f}")
    
    return vector_match

if __name__ == "__main__":
    test_u3_similarity()