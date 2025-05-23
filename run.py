import argparse
import os
import subprocess

argparser = argparse.ArgumentParser()
argparser.add_argument('--algo', default="uft", choices=["sft", "rft", "stage", "r3", "uft"], type=str)
argparser.add_argument('--n_gpu', default=1, type=int)
argparser.add_argument('--visible-devices', default="", type=str, help="Devices to Use, e.g., \"0,1,2,3\"")
argparser.add_argument('--T', default=500, type=int, help="Total Training Steps")
argparser.add_argument('--T_hint', default=300, type=int, help="Maximum Training Steps for Hint")
argparser.add_argument('--data', default="countdown", choices=["countdown", "math", "kk_logic", "others"], type=str)
argparser.add_argument('--model', default="Qwen/Qwen2.5-1.5B", type=str)
argparser.add_argument('--tp_size', default=1, type=int)
argparser.add_argument('--eval', action="store_true")
argparser.add_argument('--sft_loss_coef', default=0.001, type=float)
argparser.add_argument('--idx', default=0, type=int)
argparser.add_argument('--n_rollouts', default=4, type=int)

args = argparser.parse_args()

environment_variables = {}

environment_variables["ADV_ESTIMATOR"] = "grpo"
environment_variables["USE_KL_LOSS"] = "True"
environment_variables["ROLLOUT_N"] = f"{args.n_rollouts}"
environment_variables["SEED"] = "0"

environment_variables["N_GPUS"] = f"{args.n_gpu}"
if args.visible_devices != "":
    environment_variables["CUDA_VISIBLE_DEVICES"] = args.visible_devices
environment_variables["BASE_MODEL"] = args.model
environment_variables["ROLLOUT_TP_SIZE"] = str(args.tp_size)
environment_variables["EXPERIMENT_NAME"] = f"{args.data}-{args.model}-{args.algo}"
environment_variables["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

environment_variables["LOWER_PROB"] = f"0.05" if args.algo == "uft" else "0.0"
environment_variables["UPPER_PROB"] = f"0.95" if args.algo == "uft" else "0.0"
    
environment_variables["DATA_DIR"] = f"./data/{args.data}"
environment_variables["TOTAL_TRAINING_STEPS"] = f"{args.T}"
environment_variables["TOTAL_TRAINING_STEPS_HINT"] = f"{args.T_hint}"

environment_variables["PPO_MINIBATCH"] = "64"
environment_variables["PPO_MICROBATCH"] = "4"
environment_variables["ROLLOUT_LOGPROB_MICROBATCH"] = "4"
environment_variables["REF_LOGPROB_MICROBATCH"] = "4"
environment_variables["CRITIC_MICROBATCH"] = "4"

if args.data == "countdown":
    environment_variables["MAX_PROMPT_LENGTH"] = "256"
    environment_variables["MAX_RESPONSE_LENGTH"] = "1024"
elif args.data in ["math", "others"]:
    environment_variables["MAX_PROMPT_LENGTH"] = "1024"
    environment_variables["MAX_RESPONSE_LENGTH"] = "1024"
    if args.n_gpu == 1:
        environment_variables["PPO_MINIBATCH"] = "32"
        environment_variables["PPO_MICROBATCH"] = "2"
        environment_variables["ROLLOUT_LOGPROB_MICROBATCH"] = "2"
        environment_variables["REF_LOGPROB_MICROBATCH"] = "2"
        environment_variables["CRITIC_MICROBATCH"] = "2"
elif args.data == "kk_logic":
    environment_variables["MAX_PROMPT_LENGTH"] = "1024"
    environment_variables["MAX_RESPONSE_LENGTH"] = "1024"
else:
    raise NotImplementedError

if args.algo == "sft":
    environment_variables["SFT_MICROBATCH"] = "16"
    environment_variables["MASTER_PORT"] = f"{12345 + args.idx}"
if args.algo == "r3":
    environment_variables["UNIFORM_SAMPLING"] = "True"
    environment_variables["SFT_LOSS_COEF"] = 0.0
else:
    environment_variables["UNIFORM_SAMPLING"] = "False"
    environment_variables["SFT_LOSS_COEF"] = args.sft_loss_coef

environment_variables["STAGE"] = "True" if args.algo=="stage" else "False"

os.makedirs("temp/", exist_ok=True)
shell_dir = os.path.join("temp", f"{args.idx}.sh")

with open(shell_dir, "w") as f:
    for k in environment_variables:
        print(f"export {k}={environment_variables[k]}", file=f)

print("Environment variables set:")

print("=====================================")

os.system(f"cat {shell_dir}")

print("=====================================")

#input("Press Enter to continue...")

if args.algo == "sft":
    os.system(f"bash -c 'source {shell_dir} && exec bash scripts/sft.sh'")
elif args.eval:
    os.system(f"bash -c 'source {shell_dir} && exec bash scripts/eval.sh'")
else:
    os.system(f"bash -c 'source {shell_dir} && exec bash scripts/train.sh'")

# os.system("rm {shell_dir}")
