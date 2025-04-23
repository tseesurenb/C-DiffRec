import os
import sys

# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import data_utils  # From parent directory

def verify_exact_similarity_vector():
    """
    Test function to verify the DataDiffusion class returns EXACTLY the same
    similarity vector shown in the screenshot.
    """
    print("Verifying exact similarity vector calculation...")
    
    # Create interaction matrix exactly as shown in screenshot
    interaction_data = np.array([
        # Columns 1  2  3  4  5  6  7  8  (indexed as 0-7 in code)
        [1, 0, 1, 0, 0, 0, 1, 0],  # u_0
        [1, 1, 1, 0, 0, 1, 0, 1],  # u_1
        [0, 0, 0, 1, 1, 0, 0, 0],  # u_2
        [0, 0, 1, 1, 0, 1, 0, 1],  # u_3
        [1, 1, 1, 0, 0, 1, 1, 0],  # u_4
        [0, 0, 0, 0, 0, 1, 1, 1],  # u_5
        [1, 0, 0, 0, 0, 0, 1, 1],  # u_6
        [0, 1, 1, 1, 0, 1, 1, 0]   # u_7
    ])
    
    # Expected vector from screenshot
    expected_vector = np.array([0.671, 1.342, 1.342, 0.671, 0.000, 1.919, 1.248, 1.248])
    
    # Convert to sparse matrix as expected by the functions
    train_data = sp.csr_matrix(interaction_data)
    
    print("\n----- Step 1: Verify compute_similarity_matrix -----")
    similarity_matrix = data_utils.compute_similarity_matrix(train_data, method='cosine')
    
    # Print similarity scores for u_3 to verify against screenshot
    print("\nSimilarity scores for u_3:")
    print(f"u_3 to u_0: {similarity_matrix[3,0]:.3f} (Expected: 0.289)")
    print(f"u_3 to u_1: {similarity_matrix[3,1]:.3f} (Expected: 0.671)")
    print(f"u_3 to u_2: {similarity_matrix[3,2]:.3f} (Expected: 0.354)")
    print(f"u_3 to u_4: {similarity_matrix[3,4]:.3f} (Expected: 0.447)")
    print(f"u_3 to u_5: {similarity_matrix[3,5]:.3f} (Expected: 0.577)")
    print(f"u_3 to u_6: {similarity_matrix[3,6]:.3f} (Expected: 0.289)")
    print(f"u_3 to u_7: {similarity_matrix[3,7]:.3f} (Expected: 0.671)")
    
    print("\n----- Step 2: Verify get_top_k_similar_users -----")
    top_indices, top_similarities = data_utils.get_top_k_similar_users(similarity_matrix, 3, k=3)
    
    print(f"Top 3 similar users for u_3: {['u_' + str(idx) for idx in top_indices]}")
    print(f"With similarity scores: {[round(s, 3) for s in top_similarities]}")
    
    # Expected top users from screenshot: u_1, u_7, u_5
    expected_top_users = [1, 7, 5]
    top_users_correct = all(u in top_indices for u in expected_top_users)
    print(f"Top users match expected: {top_users_correct}")
    
    print("\n----- Step 3: Verify create_weighted_similarity_vector -----")
    # Directly call the function to check raw vector
    raw_vector = data_utils.create_weighted_similarity_vector(
        similarity_matrix, 3, top_k=3, train_data=train_data
    )
    
    print("\nComponent-wise breakdown:")
    # Vector component for u_1 (weight 0.671)
    u1_vector = train_data[1].toarray()[0] * 0.671
    print(f"u_1 contribution (weight 0.671): {np.round(u1_vector, 3)}")
    
    # Vector component for u_7 (weight 0.671)
    u7_vector = train_data[7].toarray()[0] * 0.671
    print(f"u_7 contribution (weight 0.671): {np.round(u7_vector, 3)}")
    
    # Vector component for u_5 (weight 0.577)
    u5_vector = train_data[5].toarray()[0] * 0.577
    print(f"u_5 contribution (weight 0.577): {np.round(u5_vector, 3)}")
    
    # Sum of components
    manual_sum = u1_vector + u7_vector + u5_vector
    print(f"\nManual sum of components: {np.round(manual_sum, 3)}")
    print(f"From create_weighted_similarity_vector: {np.round(raw_vector, 3)}")
    print(f"Expected from screenshot: {expected_vector}")
    
    # Check if vectors match (with small tolerance for rounding)
    functions_match = np.allclose(manual_sum, raw_vector, atol=0.001)
    expected_match = np.allclose(raw_vector, expected_vector, atol=0.001)
    
    print(f"\nManual calculation matches function output: {functions_match}")
    print(f"Function output matches expected vector: {expected_match}")
    
    print("\n----- Step 4: Verify DataDiffusion class with 'none' normalization -----")
    # Create torch tensor from interaction data
    torch_data = torch.FloatTensor(train_data.toarray())
    
    # Create DataDiffusion dataset with 'none' normalization
    dataset = data_utils.DataDiffusion(
        torch_data, 
        similarity_matrix=similarity_matrix,
        normalization_method='none'
    )
    
    # Get item for u_3
    item, similarity_vec = dataset[3]
    
    # Convert to numpy for easier comparison
    similarity_vec_np = similarity_vec.numpy()
    
    # Compare with expected vector
    print(f"DataDiffusion vector for u_3: {np.round(similarity_vec_np, 3)}")
    print(f"Expected vector from screenshot: {expected_vector}")
    
    datadiffusion_match = np.allclose(similarity_vec_np, expected_vector, atol=0.001)
    print(f"\nDataDiffusion output matches expected vector: {datadiffusion_match}")
    
    # Element-wise comparison for debugging
    if not datadiffusion_match:
        print("\nElement-wise comparison:")
        for i in range(len(expected_vector)):
            print(f"Position {i}: Expected {expected_vector[i]:.3f}, Got {similarity_vec_np[i]:.3f}, " + 
                  f"Diff: {abs(expected_vector[i] - similarity_vec_np[i]):.3f}")
    
    print("\n----- Summary of tests -----")
    if functions_match and expected_match and datadiffusion_match:
        print("✅ SUCCESS: All tests passed! The DataDiffusion class returns exactly the same vector as shown in the screenshot.")
    else:
        print("❌ FAILURE: Some tests failed.")
        if not functions_match:
            print("  - Manual calculation doesn't match function output")
        if not expected_match:
            print("  - Function output doesn't match expected vector")
        if not datadiffusion_match:
            print("  - DataDiffusion class doesn't return expected vector")
    
    return functions_match and expected_match and datadiffusion_match

if __name__ == "__main__":
    # Make sure you've imported torch.nn.functional as F in your data_utils.py file!
    verify_exact_similarity_vector()