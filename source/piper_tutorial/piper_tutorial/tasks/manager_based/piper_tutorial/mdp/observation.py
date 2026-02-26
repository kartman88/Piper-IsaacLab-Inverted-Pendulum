from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from isaaclab.envs import mdp
from isaaclab.sensors import ImuCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def quat_to_euler_xyz(quat):
    """
    Converte quaternioni (w, x, y, z) in angoli di Eulero (roll, pitch, yaw).
    Implementazione manuale per evitare errori di import.
    """
    # Normalizziamo il quaternione per sicurezza
    # quat è shape (N, 4) -> w, x, y, z
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (rotazione su X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (rotazione su Y)
    sinp = 2 * (w * y - z * x)
    # Clamp per evitare errori numerici fuori da -1/1
    sinp = torch.clamp(sinp, -1.0, 1.0) 
    pitch = torch.asin(sinp)

    # Yaw (rotazione su Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def imu_get_pole_state(env, sensor_cfg) -> torch.Tensor:
    """Get the pole state (orientation and velocity) from an IMU sensor."""
    
    # 1. Recuperiamo il sensore (dal dizionario plurale)
    sensor = env.scene.sensors[sensor_cfg.name] 
    
    # 2. Orientamento: Usa .quat_w (World Frame)
    #    ERRORE VECCHIO: sensor.data.quat_w_world
    roll, pitch, yaw = quat_to_euler_xyz(sensor.data.quat_w)
    
    # 3. Velocità: Usa .ang_vel_b (Body Frame)
    #    Gli IMU leggono la velocità angolare "sentita" dal sensore stesso.
    #    ERRORE VECCHIO: sensor.data.ang_vel_w_world
    ang_vel = sensor.data.ang_vel_b

    # 4. Concatenazione
    obs = torch.cat([
        pitch.unsqueeze(-1),      # Angolo Pitch
        ang_vel[:, 1].unsqueeze(-1) # Velocità su Y (Body frame)
    ], dim=-1)
    
    return obs

    