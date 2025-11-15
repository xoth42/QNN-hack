import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random as pyrandom

from density_qnn import (
    RBS,
    get_theta,
    RandRBS,
    string_from_RBS_connections,
    matrix_from_IRBS_string,
    density_layer,
    twopi,
)


def test_RBS_shape_and_orthogonality():
    theta = 0.3
    M = RBS(theta)
    assert M.shape == (4, 4)
    # RBS is a real orthogonal block; M^T M == I
    assert torch.allclose(M.t() @ M, torch.eye(4), atol=1e-6)


def test_get_theta_range():
    pyrandom.seed(0)
    t = get_theta()
    assert 0.0 <= t <= twopi


def test_RandRBS_deterministic_with_seed():
    # RandRBS uses the random module under the hood; seeding should make it deterministic
    pyrandom.seed(1)
    A = RandRBS()
    pyrandom.seed(1)
    B = RandRBS()
    assert torch.allclose(A, B)


def test_string_from_RBS_connections_simple():
    s = string_from_RBS_connections(((1, 2),), 3)
    # For a single connection at position 1 with 3 qubits, expected string is 'RBSI'
    assert s == "RBSI"


def test_matrix_from_I_string_identity():
    s = "III"
    M = matrix_from_IRBS_string(s)
    assert M.shape == (8, 8)
    assert torch.allclose(M, torch.eye(8), atol=1e-6)


def test_density_layer_output_shape():
    # ensure the returned density_layer function accepts weights and returns correct-sized matrix
    pyrandom.seed(2)
    qubits = 3
    matrix_count = 4
    layer_fn = density_layer(qubits, matrix_count)
    weights = torch.ones(matrix_count)
    out = layer_fn(weights)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2 ** qubits, 2 ** qubits)
