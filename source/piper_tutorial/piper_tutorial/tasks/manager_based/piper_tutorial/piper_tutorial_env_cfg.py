# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.sensors import ImuCfg

from . import mdp
from .mdp.observation import imu_get_pole_state

def set_joint_targets_to_default(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Imposta i target PD dei giunti alla loro posizione di default."""
    asset = env.scene[asset_cfg.name]
    # Estrae i valori di default (es. -2.9 per il joint3) che hai definito nell'init_state
    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    # Assegna questi valori come target per i motori
    asset.set_joint_position_target(default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

##
# Pre-defined configs
##

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@configclass
class PiperRobotCfg(ArticulationCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path=os.path.join(CURRENT_DIR, "piper_pole_tutorial.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Importante per la stabilità del braccio
            max_depenetration_velocity=10.0,
        ),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        # Posizione iniziale del robot nel mondo (x, y, z)
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0, #Arm base
            "joint2": 0.0, #Arm link 1
            "joint3": -2.9, #Arm link 2
            "joint4": 0.0, #Arm link 3
            "joint5": 0.0, #Arm link 4
            "joint7": 0.0, #Pole joint (start upright)
        },
        # !!! MODIFICA QUI: Se vuoi che il braccio parta in una posa specifica, definisci i giunti
        # joint_pos={"joint1": 0.0, "joint2": -1.57, ...}
    )
    #fix_root_link = True #The base is fixed to the gound, we only want to control the arm
    # Definiamo i motori (Actuators).
    # Isaac Lab sovrascrive i valori del USD con questi per il training
    actuators = {
        "arm_joints": ImplicitActuatorCfg(  #Use DelayedPDActuatorCfg for Sim2Real
            joint_names_expr=["joint1"], 
            effort_limit_sim=30.0, #Max effort
            stiffness=100.0, #Rigidity (K_p)
            damping=12.0, #(K_d)
            velocity_limit = 3.14, 
            #Add delay of 1/2 physics step when the type is DelayedPDActuatorCfg
            #min_delay = 1,
            #max_delay = 2,
        ),
        "static_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint2", "joint3", "joint4", "joint5"],
            effort_limit_sim=30.0,
            stiffness=400.0,      # Alta rigidità per tenerli bloccati come un blocco di marmo
            damping=40.0,
            velocity_limit = 3.14, 
        ),
    }


##
# Scene definition
##


@configclass
class PiperTutorialSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # robot
    robot: ArticulationCfg = PiperRobotCfg(prim_path="{ENV_REGEX_NS}/Robot")

    #Add IMU sensor to the scene
    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7", #Exact path of the USD
        update_period=0.01, #100Hz (match the simulation dt for simplicity)
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Robot control joint 1 to rotate the arm and stabilize the pole upright."""
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1"], 
        scale=0.25, # Scale to convert the action [-1, 1] to radiant (0.5 rad per step = 28 degrree wrt default pose)
        use_default_offset=True, #The action is relative to the default pose defined in the USD file, not absolute position
    )


@configclass
class ObservationsCfg:
    """Input for the policy network."""

    @configclass
    class PolicyCfg(ObsGroup):
        #Position and speed of the joints (relative to the default pose)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"])}
            )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"])}
            )

        #Pole state: orientation and velocity
        #Read from real IMU sensor
        pole_state = ObsTerm(
            func=imu_get_pole_state, 
            params={"sensor_cfg": SceneEntityCfg("imu_sensor")}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False #False to avoid sensor noiser
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """We choose what happen when the episode start or reset"""

    reset_static_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3", "joint4", "joint5"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    #Set the target for the motors we don't move to the default position defined in the USD file, so they stay still during the episode
    set_static_targets = EventTerm(
        func=set_joint_targets_to_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3", "joint4", "joint5"]),
        },
    )

    #Reset arm to a random position around home
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    #Reset the Pole: We let it start almost upright but with a small random offset
    #The robot learn to correct it immediatly
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            #We have to put the name of the arm pole Joint
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint7"]), 
            "position_range": (-0.05, 0.05), # +/- 0.05 radianti (circa 3 gradi)
            "velocity_range": (-0.05, 0.05),
        },
    )

    #DOMAIN RANDOMIZATION UNCOMMENT ONLY WHEN SIM2REAL
    """
    #Pole mass randomization
    randomize_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup", #Happen only once during the creation of the environment, not at every reset
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link7"]), 
            "mass_distribution_params": (-0.05, 0.05), #Add or Subtract between -50 e +50 grams
            "operation": "add", #Specify to add the random value to the base mass
        },
    )

    #Friction randomization
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]), #We apply this to the whole robot (all links)
            "static_friction_range": (0.8, 1.2), #Static friction variation (+/- 20%)
            "dynamic_friction_range": (0.8, 1.2), #Dynamic friction variation (+/- 20%)
            "restitution_range": (0.0, 0.0), #No restitution, we don't want the pole to bounce when it fall
            "num_buckets": 64, #Number of buckets for the randomization, more buckets means more variety but also more memory usage (default 64)
        },
    )

    #Apply disturbance after n seconds
    push_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="interval", #Periodically executed
        interval_range_s=(2.0, 4.0), #Random interval between 2 and 4 seconds
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint7"]),
            "position_range": (0.0, 0.0), #Keep the position unchanged
            "velocity_range": (-0.2, 0.2), #Apply random velocity disturbance
        },
    )
    """


@configclass
class RewardsCfg:
    """Teach the robot what to do"""

    #Reward to be alive (avoid robot to end rapidly the episode)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    #(Main Task) Keep the pole upright: reward is max when the poleis upright (position 0) decrease if inclined
    #Punish if the psoition is different from 0
    pole_upright = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-2.0, # Penalità alta
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint7"]), 
            "target": 0.0
        },
    )

    # Punish too rapid movements
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    #Punish is the velocity of the arm is too high (optional)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"])},
    )

    # Incoraggia il braccio a stare vicino allo 0 per non finire la corsa
    arm_centered = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.1, # Peso molto basso, il bilanciamento del palo ha priorità assoluta!
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]), 
            "target": 0.0
        },
    )


@configclass
class TerminationsCfg:
    """Game Over Conditions"""

    #Run out of time
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #Pole has fallen (Angle > 29°, around 0.5061455 rad)
    pole_fallen = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint7"]), 
            "bounds": (-0.5061455, 0.5061455) #If out of bound reset the episode
        },
    )


##
# Environment configuration
##


@configclass
class PiperTutorialEnvCfg(ManagerBasedRLEnvCfg):
    #Scene settings
    scene: PiperTutorialSceneCfg = PiperTutorialSceneCfg(num_envs=4096, env_spacing=4.0)
    #Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    #MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Configurazioni di simulazione."""
        self.decimation = 2  #Robot makes desicion every 2 physical steps (60Hz control on physics steps 120Hz)
        self.episode_length_s = 10 #Max episode Lenght in second (if not terminated before)
        
        self.viewer.eye = (3.0, 3.0, 3.0) #Camera position
        
        self.sim.dt = 1 / 120 #Physics step at 120Hz
        self.sim.render_interval = self.decimation