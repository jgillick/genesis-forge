from __future__ import annotations

import re
import torch
import numpy as np
try:
    import genesis as gs
except ModuleNotFoundError:
    print("Genesis package was not found")
    gs=None
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

if gs is None:
    # genesis helper function reimplimenetd
    def inv_quat(quat):
        if isinstance(quat, torch.Tensor):
            _quat = quat.clone()
            _quat[..., 1:].neg_()
        elif isinstance(quat, np.ndarray):
            _quat = quat.copy()
            _quat[..., 1:] *= -1
        else:
            raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
        return _quat

    def _tc_transform_by_quat(v, quat, out=None):
        if out is None:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        v_x, v_y, v_z = torch.unbind(v, dim=-1)
        q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)
        q_ww, q_wx, q_wy, q_wz = torch.unbind(q_w * quat, -1)
        q_xx, q_xy, q_xz = torch.unbind(q_x * quat[..., 1:], -1)
        q_yy, q_yz = torch.unbind(q_y * quat[..., 2:], -1)
        q_zz = q_z[..., 0] * quat[..., 3]

        out[..., 0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
        out[..., 1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
        out[..., 2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

        out /= (q_ww + q_xx + q_yy + q_zz)[..., None]

        return out

    def _np_transform_by_quat(v, quat, out=None):
        if out is None:
            out_ = np.empty(v.shape, dtype=v.dtype)
        else:
            assert out.shape == v.shape
            out_ = out

        v_T, quat_T, out_T = v.T, quat.T, out_.T
        v_x, v_y, v_z = v_T
        q_ww, q_wx, q_wy, q_wz = quat_T[0] * quat_T
        q_xx, q_xy, q_xz = quat_T[1] * quat_T[1:]
        q_yy, q_yz = quat_T[2] * quat_T[2:]
        q_zz = quat_T[3] * quat_T[3]

        out_T[0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
        out_T[1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
        out_T[2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

        out_T /= q_ww + q_xx + q_yy + q_zz

        return out_

    def transform_by_quat(v, quat):
        """
        This method transforms quat_v by quat_u.

        This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
        """
        assert v.ndim >= 1 and quat.ndim >= 1

        if all(isinstance(e, torch.Tensor) for e in (v, quat)):
            return _tc_transform_by_quat(v, quat)
        elif all(isinstance(e, np.ndarray) for e in (v, quat)):
            return _np_transform_by_quat(v, quat, out=None)
        else:
            raise TypeError(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")
else:
    transform_by_quat=gs.utils.geom.transform_by_quat
    inv_quat=gs.utils.geom.inv_quat
    

def entity_lin_vel(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's linear velocity in its local frame.

    Args:
        entity: The entity to calculate the linear velocity of

    Returns:
        torch.Tensor: Linear velocity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    return transform_by_quat(entity.get_vel(), inv_base_quat)


def entity_ang_vel(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's angular velocity in its local frame.

    Args:
        entity: The entity to calculate the angular velocity of

    Returns:
        torch.Tensor: Angular velocity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    return transform_by_quat(entity.get_ang(), inv_base_quat)


def entity_projected_gravity(entity: RigidEntity) -> torch.Tensor:
    """
    Calculate an entity's projected gravity in its local frame.

    Args:
        entity: The entity to calculate the projected gravity of

    Returns:
        torch.Tensor: Projected gravity in the local frame
    """
    inv_base_quat = inv_quat(entity.get_quat())
    gravity = torch.tensor(
        [0.0, 0.0, -1.0], 
        device=inv_base_quat.device, 
        dtype=inv_base_quat.dtype
    ).expand(inv_base_quat.shape[0], 3)
    return transform_by_quat(gravity, inv_base_quat)


def links_by_name_pattern(entity: RigidEntity, name_pattern: str) -> list[RigidLink]:
    """
    Find a list of entity links by name regex pattern.

    Args:
        entity: The entity to find the links in.
        name_re: The name regex patterns of the links to find.

    Returns:
        List of RigidLink objects.
    """
    links = []
    for link in entity.links:
        if link.name == name_pattern or re.match(f"^{name_pattern}$", link.name):
            links.append(link)
    return links
