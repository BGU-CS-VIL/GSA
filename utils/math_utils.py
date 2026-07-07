import numpy as np
import torch

def quaternion_multiply(q1, q2):
    """
    Batch quaternion multiplication.
    
    Args:
    q1 (torch.Tensor): First set of quaternions (N, 4)
    q2 (torch.Tensor): Rotation quaternion (4)
    
    Returns:
    torch.Tensor: Multiplied quaternions (N, 4)
    """
    if q2.ndim == 1:
        q2 = q2.unsqueeze(0)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=1)

def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    return torch.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
    ])

# rotation matrix to quaternion
def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion"""
    r = R.ravel()
    q = np.zeros(4)
    q[0] = 0.5 * np.sqrt(1 + r[0] + r[4] + r[8])
    q[1] = (r[7] - r[5]) / (4 * q[0])
    q[2] = (r[2] - r[6]) / (4 * q[0])
    q[3] = (r[3] - r[1]) / (4 * q[0])
    return q


def build_scaling_rotation(scaling, rotation):
    """Convert scale and rotation to 3x3 matrix L where covariance = L @ L.T"""
    L = np.zeros((len(scaling), 3, 3))
    L[:, 0, 0] = scaling[:, 0]
    L[:, 1, 1] = scaling[:, 1]
    L[:, 2, 2] = scaling[:, 2]

    for i in range(len(rotation)):
        R = quaternion_to_rotation_matrix(rotation[i])
        L[i] = R @ L[i]

    return L