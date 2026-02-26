# step_test_si.py

import argparse
import threading  # !!! Aggiunto per leggere la tastiera senza bloccare il simulatore !!!
from isaaclab.app import AppLauncher

# 1. Inizializzazione standard
parser = argparse.ArgumentParser(description="Test Step Response per Piper")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import piper_tutorial.tasks  

# Variabile globale che farà da "interruttore"
start_movement = False

def wait_for_enter():
    """Questa funzione gira in background e aspetta il tasto INVIO"""
    global start_movement
    input("\n[ATTESA] Premi INVIO qui nel terminale per far scattare il braccio...\n")
    start_movement = True

def main():
    env_cfg = parse_env_cfg("Template-Piper-Tutorial-v0")
    env_cfg.scene.num_envs = 1          
    env_cfg.episode_length_s = 10000.0   
    
    if hasattr(env_cfg.terminations, "pole_fallen"):
        env_cfg.terminations.pole_fallen = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("=====================================================")
    print("[TEST AVVIATO] La simulazione è attiva in tempo reale.")
    print("[TEST AVVIATO] Sistema la telecamera a tuo piacimento.")
    print("=====================================================")

    dt = env.step_dt 
    obs, _ = env.reset()
    step_count = 0
    
    robot = env.scene["robot"]
    joint_idx = robot.find_joints("joint1")[0][0] 
    
    action_sent = False
    reached_target = False
    start_time = 0.0
    initial_pos = 0.0

    # !!! Avviamo la lettura della tastiera in background !!!
    threading.Thread(target=wait_for_enter, daemon=True).start()

    while simulation_app.is_running():
        actions = torch.zeros((env.num_envs, 1), device=env.device)
        
        # Finché non premi INVIO, aggiorniamo la posizione iniziale e stiamo fermi
        if not start_movement:
            initial_pos = robot.data.joint_pos[0, joint_idx].item()
        else:
            # Hai premuto INVIO! Spariamo l'azione al massimo
            actions[:, 0] = 1.0 
            
            # Facciamo partire il cronometro solo la prima volta
            if not action_sent:
                action_sent = True
                start_time = step_count * dt
                print(f"[{start_time:.2f}s] Comando inviato! Inizio misurazione tempo...")
        
        # Mandiamo l'azione al simulatore
        obs, *rest = env.step(actions)
        
        # Logica del Cronometro Intelligente
        if action_sent and not reached_target:
            current_pos = robot.data.joint_pos[0, joint_idx].item()
            current_vel = robot.data.joint_vel[0, joint_idx].item()
            
            # Condizione di arrivo: 
            # 1. Si è spostato di almeno 0.05 radianti
            # 2. La sua velocità è scesa sotto 0.05 rad/s (si è "fermato")
            if abs(current_pos - initial_pos) > 0.05 and abs(current_vel) < 0.05:
                end_time = step_count * dt
                time_taken = end_time - start_time
                print(f"[{end_time:.2f}s] BERSAGLIO RAGGIUNTO E STABILIZZATO!")
                print("=====================================================")
                print(f"[RISULTATO] Tempo di risposta (Settling Time): {time_taken:.3f} secondi")
                print("=====================================================")
                reached_target = True # Stoppa il cronometro per non stampare all'infinito

        step_count += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()