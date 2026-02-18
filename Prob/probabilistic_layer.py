"""
Probabilistic Semantic Alignment Layer
=======================================

This module implements the core probabilistic upgrade from CLIP's cosine similarity
to a distribution-based semantic matching system.

Key Innovation:
- Instead of representing each emoji as a point vector, we represent it as a 
  probability distribution (Gaussian) in the embedding space
- Text embeddings are evaluated under each emoji's distribution
- Probabilities are normalized globally across all emojis

Mathematical Foundation:
    p(e_i | z) = α_i * p(z | μ_i, Σ_i, β_i)
    P(e_i | z) = p(e_i | z) / Σ_j p(e_j | z) 

where:
    z: text embedding from encoder
    μ_i: mean embedding for emoji i (semantic center)
    Σ_i: covariance matrix for emoji i (semantic spread)
    β_i: temperature parameter (controls sharpness)
    α_i: mixture weight (global normalization factor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GaussianEmojiDistribution(nn.Module):
    """
    Represents a single emoji as a Gaussian distribution in embedding space.
    
    This is the fundamental building block that replaces CLIP's point embeddings
    with probabilistic representations.
    
    Parameters:
        embedding_dim: Dimension of the shared embedding space
        use_full_covariance: If True, use full covariance matrix. 
                            If False, use diagonal (computationally cheaper)
        init_std: Initial standard deviation for the distribution
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        use_full_covariance: bool = False,
        init_std: float = 1.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_full_covariance = use_full_covariance
        
        # μ: Mean embedding (semantic center)
        # Initialize with small random values, will be learned during training
        self.mu = nn.Parameter(torch.randn(embedding_dim) * 0.01)
        
        # Σ: Covariance parameters (semantic spread)
        if use_full_covariance:
            # Full covariance: more expressive but O(d²) parameters
            # Store as Cholesky decomposition for numerical stability
            self.L = nn.Parameter(torch.eye(embedding_dim) * init_std)
        else:
            # Diagonal covariance: O(d) parameters, assumes feature independence
            # Store log(σ²) to ensure positivity via exp transformation
            self.log_var = nn.Parameter(torch.ones(embedding_dim) * math.log(init_std ** 2))
        
        # β: Temperature parameter (controls sharpness of distribution)
        # Higher β → sharper distribution (more confident)
        # Lower β → flatter distribution (more uncertain)
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # Init to β=1
        
    def get_covariance(self) -> torch.Tensor:
        """Get the covariance matrix Σ."""
        if self.use_full_covariance:
            # Σ = L * L^T (ensures positive semi-definite)
            return torch.matmul(self.L, self.L.t())
        else:
            # Diagonal covariance
            return torch.diag(torch.exp(self.log_var))
    
    def get_precision(self) -> torch.Tensor:
        """Get the precision matrix Σ^(-1) (inverse of covariance)."""
        if self.use_full_covariance:
            # For numerical stability, use Cholesky solve
            return torch.cholesky_inverse(self.L)
        else:
            # Diagonal precision is just 1/variance
            return torch.diag(torch.exp(-self.log_var))
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of text embedding z under this emoji's distribution.
        
        Log-space computation for numerical stability.
        
        Args:
            z: Text embeddings of shape (batch_size, embedding_dim) or (embedding_dim,)
            
        Returns:
            Log probability of shape (batch_size,) or scalar
        """
        # Ensure z is at least 2D
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        batch_size = z.shape[0]
        
        # Difference from mean: (z - μ)
        diff = z - self.mu  # Shape: (batch_size, embedding_dim)
        
        # Get precision matrix and temperature
        precision = self.get_precision()
        beta = torch.exp(self.log_beta)
        
        # Compute Mahalanobis distance: (z - μ)^T Σ^(-1) (z - μ)
        if self.use_full_covariance:
            # Full covariance case
            mahalanobis = torch.sum(diff @ precision * diff, dim=1)
        else:
            # Diagonal covariance case (more efficient)
            mahalanobis = torch.sum(diff ** 2 * torch.exp(-self.log_var), dim=1)
        
        # Scale by temperature
        mahalanobis = beta * mahalanobis
        
        # Compute log probability
        # log p(z|μ,Σ,β) = -0.5 * [β * (z-μ)^T Σ^(-1) (z-μ) + d*log(2π) - log(det(Σ)) - d*log(β)]
        if self.use_full_covariance:
            log_det_cov = 2 * torch.sum(torch.log(torch.diag(self.L)))
        else:
            log_det_cov = torch.sum(self.log_var)
        
        log_prob = -0.5 * (
            mahalanobis + 
            self.embedding_dim * math.log(2 * math.pi) - 
            beta * log_det_cov -
            self.embedding_dim * self.log_beta
        )
        
        return log_prob


class ProbabilisticEmojiEncoder(nn.Module):
    """
    Probabilistic encoder for all emojis.
    
    This replaces CLIP's emoji embedding matrix with a collection of 
    Gaussian distributions, one per emoji.
    
    Key difference from CLIP:
    - CLIP: num_emojis × embedding_dim matrix
    - This: num_emojis × GaussianDistribution objects
    
    Parameters:
        num_emojis: Number of emojis in the vocabulary
        embedding_dim: Dimension of shared embedding space
        use_full_covariance: Whether to use full or diagonal covariance
        init_std: Initial standard deviation for distributions
    """
    
    def __init__(
        self,
        num_emojis: int,
        embedding_dim: int,
        use_full_covariance: bool = False,
        init_std: float = 1.0
    ):
        super().__init__()
        
        self.num_emojis = num_emojis
        self.embedding_dim = embedding_dim
        
        # Create one Gaussian distribution per emoji
        self.emoji_distributions = nn.ModuleList([
            GaussianEmojiDistribution(
                embedding_dim=embedding_dim,
                use_full_covariance=use_full_covariance,
                init_std=init_std
            )
            for _ in range(num_emojis)
        ])
        
        # α: Global normalization weights (learned)
        # These reflect the relative frequency/importance of each emoji
        # Initialized uniformly, will adapt during training
        self.log_alpha = nn.Parameter(torch.zeros(num_emojis))
    
    def compute_log_likelihoods(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(e_i | z) for all emojis.
        
        Args:
            z: Text embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Log likelihoods of shape (batch_size, num_emojis)
        """
        batch_size = z.shape[0]
        log_likelihoods = torch.zeros(batch_size, self.num_emojis, device=z.device)
        
        for i, emoji_dist in enumerate(self.emoji_distributions):
            # Compute log p(z | emoji_i)
            log_prob_z = emoji_dist.log_prob(z)
            
            # Add log(α_i) to get log p(emoji_i | z) ∝ log(α_i) + log p(z | emoji_i)
            log_likelihoods[:, i] = self.log_alpha[i] + log_prob_z
        
        return log_likelihoods
    
    def compute_probabilities(
        self, 
        z: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute normalized probabilities P(e_i | z) for all emojis.
        
        This is the key replacement for cosine similarity in CLIP.
        Instead of geometric similarity, we compute probabilistic compatibility.
        
        Args:
            z: Text embeddings of shape (batch_size, embedding_dim)
            temperature: Optional temperature scaling for softmax (default: 1.0)
            
        Returns:
            Probabilities of shape (batch_size, num_emojis) that sum to 1
        """
        # Compute log likelihoods
        log_likelihoods = self.compute_log_likelihoods(z)
        
        # Global normalization via log-sum-exp (numerically stable softmax)
        # P(e_i | z) = exp(log p(e_i | z)) / Σ_j exp(log p(e_j | z))
        log_probs = log_likelihoods / temperature
        probs = F.softmax(log_probs, dim=1)
        
        return probs
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute both log-likelihoods and normalized probabilities.
        
        Args:
            z: Text embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Tuple of:
                - log_likelihoods: shape (batch_size, num_emojis)
                - probabilities: shape (batch_size, num_emojis), sums to 1
        """
        log_likelihoods = self.compute_log_likelihoods(z)
        probs = F.softmax(log_likelihoods, dim=1)
        
        return log_likelihoods, probs


# ============================================================================
# Comparison utilities to understand the difference from CLIP
# ============================================================================

def cosine_similarity_clip_style(
    text_embeddings: torch.Tensor,
    emoji_embeddings: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Traditional CLIP-style cosine similarity for comparison.
    
    Args:
        text_embeddings: (batch_size, embedding_dim)
        emoji_embeddings: (num_emojis, embedding_dim)
        temperature: Temperature parameter τ (default 0.07 as in CLIP)
        
    Returns:
        Similarity scores of shape (batch_size, num_emojis)
    """
    # Normalize embeddings
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    emoji_embeddings = F.normalize(emoji_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = text_embeddings @ emoji_embeddings.t()
    
    # Scale by temperature
    logits = similarity / temperature
    
    return logits


if __name__ == "__main__":
    # Demo: Compare probabilistic approach with CLIP-style cosine similarity
    
    print("=" * 70)
    print("Probabilistic CLIP Upgrade Demo")
    print("=" * 70)
    
    # Setup
    batch_size = 4
    num_emojis = 10
    embedding_dim = 512
    
    # Create dummy text embeddings
    text_embeddings = torch.randn(batch_size, embedding_dim)
    
    print(f"\nSetup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of emojis: {num_emojis}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    # ---- Traditional CLIP approach ----
    print("\n" + "─" * 70)
    print("Traditional CLIP (Cosine Similarity)")
    print("─" * 70)
    
    emoji_embeddings_clip = torch.randn(num_emojis, embedding_dim)
    clip_logits = cosine_similarity_clip_style(text_embeddings, emoji_embeddings_clip)
    clip_probs = F.softmax(clip_logits, dim=1)
    
    print(f"Output shape: {clip_probs.shape}")
    print(f"First sample probabilities:\n{clip_probs[0]}")
    print(f"Sum of probabilities: {clip_probs[0].sum().item():.6f}")
    
    # ---- Probabilistic approach ----
    print("\n" + "─" * 70)
    print("Probabilistic Approach (Gaussian Distributions)")
    print("─" * 70)
    
    prob_encoder = ProbabilisticEmojiEncoder(
        num_emojis=num_emojis,
        embedding_dim=embedding_dim,
        use_full_covariance=False  # Start with diagonal for efficiency
    )
    
    log_likelihoods, prob_probs = prob_encoder(text_embeddings)
    
    print(f"Output shape: {prob_probs.shape}")
    print(f"First sample probabilities:\n{prob_probs[0]}")
    print(f"Sum of probabilities: {prob_probs[0].sum().item():.6f}")
    
    # ---- Key differences ----
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    
    print("\n1. Representation:")
    print(f"   CLIP: Each emoji is a {embedding_dim}D point")
    print(f"   Probabilistic: Each emoji is a {embedding_dim}D Gaussian distribution")
    
    print("\n2. Similarity Measure:")
    print("   CLIP: Cosine angle between vectors")
    print("   Probabilistic: Likelihood under distribution")
    
    print("\n3. Parameters per emoji:")
    clip_params = embedding_dim
    prob_params = embedding_dim + embedding_dim + 1 + 1  # μ + var + β + α
    print(f"   CLIP: {clip_params} (just the embedding)")
    print(f"   Probabilistic: {prob_params} (μ, Σ, β, α)")
    
    print("\n4. Semantic flexibility:")
    print("   CLIP: Fixed 'width' of meaning for all emojis")
    print("   Probabilistic: Each emoji can have different 'spread' (via Σ)")
    
    # Demonstrate semantic spread
    print("\n" + "─" * 70)
    print("EXAMPLE: Semantic Spread Flexibility")
    print("─" * 70)
    
    emoji_0_std = torch.exp(0.5 * prob_encoder.emoji_distributions[0].log_var).mean()
    emoji_1_std = torch.exp(0.5 * prob_encoder.emoji_distributions[1].log_var).mean()
    
    print(f"Emoji 0 average std: {emoji_0_std.item():.4f}")
    print(f"Emoji 1 average std: {emoji_1_std.item():.4f}")
    print("\nInterpretation:")
    print("  - Wider distribution → emoji has broader, more ambiguous meaning")
    print("  - Narrower distribution → emoji has specific, precise meaning")
    
    print("\n" + "=" * 70)
    print("Setup complete! Ready for training implementation.")
    print("=" * 70)
