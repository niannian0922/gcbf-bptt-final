import argparse
import yaml
import torch
from gcbfplus.env.single_double_integrator import SingleDoubleIntegratorEnv
from gcbfplus.trainer.bptt_trainer import BPTTTrainer

def main():
    parser = argparse.ArgumentParser(description="Train a single agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cpu'))
    env_cfg = config.get('env', {})
    trainer_cfg = config.get('trainer', {})
    policy_cfg = config.get('policy', {})

    print(f"Using device: {device}")

    # Pass the device to the environment constructor
    env = SingleDoubleIntegratorEnv(env_cfg, device=device)
    # The trainer and policy will be created inside the BPTTTrainer
    trainer = BPTTTrainer(env, trainer_cfg, policy_cfg, device=device, full_config=config)

    print("Starting BPTT training with configuration:")
    print(f"  Run name: {trainer_cfg.get('run_name', 'default')}")
    print(f"  Steps: {trainer_cfg.get('trainer', {}).get('num_steps', 0)}")
    print(f"  Horizon: {trainer_cfg.get('bptt', {}).get('horizon_length', 0)}")
    print(f"  Log dir: {trainer_cfg.get('log_dir', 'logs')}/{trainer_cfg.get('run_name', 'default')}")

    trainer.train()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

