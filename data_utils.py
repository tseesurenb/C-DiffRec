import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

from scipy.sparse import csr_matrix
# Top-k approach (more memory efficient)
from scipy.sparse import lil_matrix
import numpy as np
        
from sklearn.metrics.pairwise import cosine_similarity


def data_load(train_path, valid_path, test_path, similarity_matrix_path=None):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    # Load similarity matrix if provided
    similarity_matrix = None
    if similarity_matrix_path is not None and os.path.exists(similarity_matrix_path):
        try:
            similarity_matrix = np.load(similarity_matrix_path)
            print(f'Similarity matrix loaded with shape: {similarity_matrix.shape}')
            
            # Verify dimensions match
            if similarity_matrix.shape[0] != n_user:
                print(f"Warning: Similarity matrix user dimension mismatch: {similarity_matrix.shape[0]} vs {n_user}")
        except Exception as e:
            print(f"Error loading similarity matrix: {e}")
            similarity_matrix = None
    
    return train_data, valid_y_data, test_y_data, n_user, n_item, similarity_matrix



def compute_similarity_matrix_dense(train_data, method='cosine'):
    """
    Efficient similarity computation using sparse operations.
    
    :param train_data: User-item interaction matrix (csr_matrix)
    :param method: 'cosine' or 'jaccard'
    :return: User similarity matrix (dense numpy array)
    """
    if method == 'cosine':
        # Use sklearn’s efficient cosine similarity on sparse matrix
        return cosine_similarity(train_data)

    elif method == 'jaccard':
        # Binary version of the matrix (ensure only 0/1 values)
        bin_data = train_data.copy()
        bin_data.data = np.ones_like(bin_data.data)

        # Compute intersection: A * A^T
        intersection = bin_data.dot(bin_data.T).tocoo()

        # Compute row sums (number of items per user)
        row_sums = np.array(bin_data.sum(axis=1)).flatten()

        # Compute union for each pair: |A| + |B| - |A ∩ B|
        union = row_sums[:, None] + row_sums[None, :] - intersection.toarray()

        # Avoid division by zero
        union[union == 0] = 1e-10

        # Compute Jaccard similarity
        similarity = intersection.toarray() / union

        return similarity

    else:
        raise ValueError(f"Unsupported similarity method: {method}")
    
from scipy.sparse import csr_matrix
import heapq

def compute_similarity_matrix(train_data, method='cosine', top_k=3):
    """
    Efficient similarity computation using sparse operations.
    
    :param train_data: User-item interaction matrix (csr_matrix)
    :param method: 'cosine' or 'jaccard'
    :param top_k: Top-K similar users to keep (only applies to jaccard)
    :return: User similarity matrix (dense for cosine, sparse dict for jaccard)
    """
    if method == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(train_data)

    elif method == 'jaccard':
        # Ensure data is binary
        bin_data = train_data.copy()
        bin_data.data = np.ones_like(bin_data.data)
        
        # Compute row sums (number of items per user)
        row_sums = np.array(bin_data.sum(axis=1)).flatten()
        
        # Compute dot product for intersections (sparse operation)
        intersection = bin_data.dot(bin_data.T)
        
        # Create result container
        n_users = bin_data.shape[0]
        
        if top_k is None:
            # Full similarity matrix approach (memory intensive but faster)
            # Convert to COO format for easier manipulation
            intersection_coo = intersection.tocoo()
            
            # Create arrays for sparse matrix construction
            rows, cols, data = [], [], []
            
            # Process each non-zero element
            for i, j, inter_val in zip(intersection_coo.row, intersection_coo.col, intersection_coo.data):
                # Skip self-similarity if desired
                if i == j:
                    continue
                    
                # Calculate union and similarity
                union_val = row_sums[i] + row_sums[j] - inter_val
                
                # Avoid division by zero
                if union_val == 0:
                    continue
                    
                sim = inter_val / union_val
                
                # Save result
                rows.append(i)
                cols.append(j)
                data.append(sim)
            
            # Create sparse similarity matrix
            from scipy.sparse import csr_matrix
            sim_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_users))
            return sim_matrix
            
        else:
            
            # Use LIL format for efficient row manipulation
            sim_matrix = lil_matrix((n_users, n_users))
            
            for i in range(n_users):
                # Get intersection values for user i with all other users
                row_intersection = intersection[i].toarray().flatten()
                
                # Calculate all unions and similarities at once
                unions = row_sums[i] + row_sums - row_intersection
                
                # Avoid division by zero
                mask = unions > 0
                sims = np.zeros_like(unions, dtype=float)
                sims[mask] = row_intersection[mask] / unions[mask]
                
                # Set self-similarity to 0
                sims[i] = 0
                
                # Get top-k indices
                if top_k < n_users:
                    top_indices = np.argpartition(sims, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(-sims[top_indices])]
                else:
                    top_indices = np.argsort(-sims)
                
                # Store only top-k similarities
                for j, idx in enumerate(top_indices):
                    if sims[idx] > 0:
                        sim_matrix[i, idx] = sims[idx]
            
            # Convert to CSR for efficient usage
            return sim_matrix.tocsr()
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")



def get_top_k_similar_users(similarity_matrix, user_id, k=3):
    """
    Find top-k similar users for a given user
    
    :param similarity_matrix: User similarity matrix (dense or sparse)
    :param user_id: Target user ID
    :param k: Number of similar users to return
    :return: Indices and similarity scores of top-k similar users
    """
    n_users = similarity_matrix.shape[0]
    
    if user_id >= n_users:
        raise IndexError(f"user_id {user_id} is out of bounds for similarity matrix with {n_users} users.")

    # Convert row to dense vector
    if sp.issparse(similarity_matrix):
        user_similarities = similarity_matrix[user_id].toarray().flatten()
    else:
        user_similarities = similarity_matrix[user_id].copy()
    
    # Exclude self-similarity
    user_similarities[user_id] = -1
    
    # Get top-k indices
    top_indices = np.argsort(user_similarities)[::-1][:k]
    top_similarities = user_similarities[top_indices]
    
    return top_indices, top_similarities


def get_top_k_similar_users_old(similarity_matrix, user_id, k=3):
    """
    Find top-k similar users for a given user
    
    :param similarity_matrix: User similarity matrix
    :param user_id: Target user ID
    :param k: Number of similar users to return
    :return: Indices and similarity scores of top-k similar users
    """
    # Get similarities for the user (excluding self)
    user_similarities = similarity_matrix[user_id].copy()
    user_similarities[user_id] = -1  # Exclude self
    
    # Get top-k indices
    top_indices = np.argsort(user_similarities)[::-1][:k]
    top_similarities = user_similarities[top_indices]
    
    return top_indices, top_similarities


def create_weighted_similarity_vector(similarity_matrix, user_id, top_k=3, train_data=None):
    """
    Create a weighted similarity vector based on top-k similar users' interaction patterns
    
    :param similarity_matrix: User similarity matrix (user-to-user)
    :param user_id: Target user ID
    :param top_k: Number of similar users to consider
    :param train_data: User-item interaction data (needed to get item vectors)
    :return: Weighted similarity vector with shape (n_items,)
    """
    # Get top-k similar users based on user-to-user similarity
    top_indices, top_similarities = get_top_k_similar_users(similarity_matrix, user_id, k=top_k)
    
    # Check if train_data is provided
    if train_data is None:
        # If no train_data is provided but similarity_matrix has correct dimensions
        # (this could be the case if similarity_matrix is already user-to-item)
        if len(similarity_matrix.shape) == 2 and similarity_matrix.shape[0] == similarity_matrix.shape[1]:
            # We're dealing with a square user-user matrix, can't create item vector without train_data
            raise ValueError("Train data must be provided when using a user-to-user similarity matrix")
        else:
            # We might have a non-square matrix that's already user-to-item
            n_items = similarity_matrix.shape[1]
            final_vector = np.zeros(n_items)
            for idx, sim_score in zip(top_indices, top_similarities):
                user_vector = similarity_matrix[idx]
                final_vector += sim_score * user_vector
            return final_vector
    
    # If train_data is provided, use it to get item interactions
    # Get train_data as dense array for easier manipulation
    if isinstance(train_data, sp.spmatrix):
        train_array = train_data.toarray()
    else:
        train_array = train_data
    
    n_items = train_array.shape[1]
    final_vector = np.zeros(n_items)
    
    # For each similar user, add their item vector weighted by similarity
    for idx, sim_score in zip(top_indices, top_similarities):
        user_vector = train_array[idx]
        final_vector += sim_score * user_vector
    
    return final_vector


class DataDiffusion(Dataset):
    def __init__(self, data, similarity_matrix=None, normalization_method='none'):
        self.data = data
        self.similarity_matrix = similarity_matrix
        self.normalization_method = normalization_method
        
    def __getitem__(self, index):
        item = self.data[index]
        
        # If similarity matrix is provided, return it with the data
        if self.similarity_matrix is not None:
            # If similarity matrix is user x user, transform it to the expected shape
            # if self.similarity_matrix.shape[1] != item.shape[0]:  # Check if dimensions match
            #     # Create a weighted recommendation vector based on similar users
            #     similarity_vector = create_weighted_similarity_vector(
            #         self.similarity_matrix, index, top_k=3, train_data=self.data.numpy()
            #     )
            #     similarity_vector = torch.FloatTensor(similarity_vector)
            # else:
            #     similarity_vector = torch.FloatTensor(self.similarity_matrix[index])

            # If similarity matrix is user x user, transform it to item space
            if self.similarity_matrix.shape[0] == self.similarity_matrix.shape[1]:  # Check if it's a square matrix (user x user)
                # Create a weighted recommendation vector based on similar users
                similarity_vector = create_weighted_similarity_vector(
                    self.similarity_matrix, index, top_k=3, train_data=self.data.numpy()
                )
                similarity_vector = torch.FloatTensor(similarity_vector)
            else:
                # It's already user x item, just use the row directly
                similarity_vector = torch.FloatTensor(self.similarity_matrix[index])
            
            # Apply normalization if needed
            if self.normalization_method == 'min_max':
                min_val = similarity_vector.min()
                max_val = similarity_vector.max()
                if max_val > min_val:
                    similarity_vector = (similarity_vector - min_val) / (max_val - min_val)
            elif self.normalization_method == 'l2':
                norm = torch.norm(similarity_vector, p=2)
                if norm > 0:
                    similarity_vector = similarity_vector / norm
            elif self.normalization_method == 'softmax':
                # Apply softmax to make the vector sum to 1, highlighting strong similarities
                # Adding a temperature parameter (default=1.0) for controlling softmax sharpness
                temperature = 1.0  # Lower values make the distribution more peaked
                similarity_vector = F.softmax(similarity_vector / temperature, dim=0)
            elif self.normalization_method == 'sigmoid':
                # Apply sigmoid to map all values to range (0,1)
                similarity_vector = torch.sigmoid(similarity_vector)
            elif self.normalization_method == 'tanh':
                # Apply tanh to map values to (-1,1) range
                similarity_vector = torch.tanh(similarity_vector)
            elif self.normalization_method == 'log':
                # Apply log normalization (with offset to handle zeros)
                eps = 1e-10
                similarity_vector = torch.log(similarity_vector + eps)
                # Rescale log values to (0,1) range
                min_val = similarity_vector.min()
                max_val = similarity_vector.max()
                if max_val > min_val:
                    similarity_vector = (similarity_vector - min_val) / (max_val - min_val)
            
            return item, similarity_vector
        
        return item
    
    def __len__(self):
        return len(self.data)