import sys, os
import argparse
import subprocess
import json
import hashlib
import itertools
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize NVML
try:
    nvmlInit()
except Exception as e:
    print(f"Failed to initialize NVML: {e}. GPU monitoring will not be available.")

# --- Constants ---
EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
SUCCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}."

# --- Helper Functions ---
def md5(key):
    return hashlib.md5(key.encode()).hexdigest()

def parse_dispatcher_config(config):
    assert all(
        [
            k
            in [
                "script",
                "available_gpus",
                "cartesian_hyperparams",
                "paired_hyperparams",
                "tune_hyperparams",
            ]
            for k in config.keys()
        ]
    )

    cartesian_hyperparamss = config["cartesian_hyperparams"]
    paired_hyperparams = config.get("paired_hyperparams", [])
    flags = []
    arguments = []
    experiment_axies = []

    # add anything outside search space as fixed
    fixed_args = ""
    for arg in config:
        if arg not in [
            "script",
            "cartesian_hyperparams",
            "paired_hyperparams",
            "available_gpus",
        ]:
            if type(config[arg]) is bool:
                if config[arg]:
                    fixed_args += "--{} ".format(str(arg))
                else:
                    continue
            else:
                fixed_args += "--{} {} ".format(arg, config[arg])

    # add paired combo of search space
    paired_args_list = [""]
    if len(paired_hyperparams) > 0:
        paired_args_list = []
        paired_keys = list(paired_hyperparams.keys())
        paired_vals = list(paired_hyperparams.values())
        flags.extend(paired_keys)
        for paired_combo in zip(*paired_vals):
            paired_args = ""
            for i, flg_value in enumerate(paired_combo):
                if type(flg_value) is bool:
                    if flg_value:
                        paired_args += "--{} ".format(str(paired_keys[i]))
                    else:
                        continue
                else:
                    paired_args += "--{} {} ".format(
                        str(paired_keys[i]), str(flg_value)
                    )
            paired_args_list.append(paired_args)

    # add every combo of search space
    product_flags = []
    for key, value in cartesian_hyperparamss.items():
        flags.append(key)
        product_flags.append(key)
        arguments.append(value)
        if len(value) > 1:
            experiment_axies.append(key)

    experiments = []
    exps_combs = list(itertools.product(*arguments))

    for tpl in exps_combs:
        exp = ""
        for idx, flg in enumerate(product_flags):
            if type(tpl[idx]) is bool:
                if tpl[idx]:
                    exp += "--{} ".format(str(flg))
                else:
                    continue
            else:
                exp += "--{} {} ".format(str(flg), str(tpl[idx]))
        exp += fixed_args
        for paired_args in paired_args_list:
            experiments.append(exp + paired_args)

    return experiments, flags, experiment_axies


def gpu_memory_available(gpu_index):
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free
    except Exception:
        # If NVML fails, return a large number to effectively disable memory checking
        return float('inf')

def run_job(job_info):
    """
    Worker function. It runs one job and returns the result.
    """
    flag_string, gpu_index, args = job_info
    script = args.script
    log_dir = args.log_dir
    dry_run = args.dry_run
    memory_threshold_bytes = args.memory_threshold * 1024 * 1024
    job_timeout = args.job_timeout  # New timeout argument

    job_hash = md5(flag_string)
    print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Job received.")

    # --- Wait for GPU Memory ---
    while True:
        free_mem = gpu_memory_available(gpu_index)
        if free_mem >= memory_threshold_bytes:
            print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Memory OK ({free_mem/1024**2:.0f}MB free). Launching.")
            break
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Waiting for memory. Needs {args.memory_threshold}MB, has {free_mem/1024**2:.0f}MB.")
        time.sleep(10)

    # --- Launch Experiment ---
    log_stem = os.path.join(log_dir, job_hash)
    log_path = f"{log_stem}.txt"
    results_path = f"{log_stem}.args" 

    experiment_string = (
        f"CUDA_VISIBLE_DEVICES={gpu_index} python -u scripts/{script}.py {flag_string} "
        f"--results_path {log_stem} --experiment_name {job_hash}"
    )

    pipe_str = ">>" if "--resume" in flag_string else ">"
    shell_cmd = f"{experiment_string} {pipe_str} {log_path} 2>&1"
    
    if dry_run:
        print(f"DRY RUN CMD: {shell_cmd}")
        return results_path, log_path, "dry_run"

    if os.path.exists(results_path):
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Result file already exists. Skipping.")
        return results_path, log_path, "skipped"

    try:
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Starting subprocess.")
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Executing command: {shell_cmd}") # <-- ADDED THIS LINE
        
        # Use Popen for non-blocking call with timeout
        process = subprocess.Popen(shell_cmd, shell=True)
        process.wait(timeout=job_timeout)  # Wait for the process to complete with a timeout

        if process.returncode == 0:
            print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: Subprocess finished successfully.")
            return results_path, log_path, "success"
        else:
            print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: CRASH! Subprocess failed with return code {process.returncode}.")
            return results_path, log_path, "crashed"

    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: TIMEOUT! Job exceeded {job_timeout} seconds. Killing process.")
        process.kill()  # Forcefully kill the runaway process
        return results_path, log_path, "timeout"
    except Exception as e:
        print(f"[GPU {gpu_index} | Job {job_hash[:6]}]: An unexpected error occurred: {e}")
        return results_path, log_path, "error"

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dispatcher.")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="path to model configurations json file")
    parser.add_argument("-l", "--log_dir", type=str, required=True, help="path to store logs and detailed job level result files")
    parser.add_argument("-n", "--dry_run", action="store_true", help="print out commands without running")
    parser.add_argument("-e", "--eval_train_config", action="store_true", help="create evaluation run from a training config")
    parser.add_argument("--max_jobs_per_gpu", type=int, default=1, help="Number of jobs to run concurrently on each GPU.")
    parser.add_argument("--memory_threshold", type=int, default=500, help="Minimum free GPU memory in MB to run a job.")
    parser.add_argument("--job_timeout", type=int, default=3600, help="Maximum time in seconds a single job can run before being killed.")
    
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.config_path))
        sys.exit(1)
    
    os.makedirs(args.log_dir, exist_ok=True)
    experiment_config = json.load(open(args.config_path, "r"))
    
    # Add script name to args object so worker can access it
    args.script = experiment_config.get("script", "main")

    experiments, _, _ = parse_dispatcher_config(experiment_config)
    
    # Create a list of all jobs to run
    # Each job is a tuple: (flag_string, gpu_index, args_object)
    gpu_cycle = itertools.cycle(experiment_config["available_gpus"])
    all_jobs = [(exp, next(gpu_cycle), args) for exp in experiments]

    print(f"Dispatcher starting with {len(all_jobs)} jobs.")
    print(f"Max concurrent jobs per GPU: {args.max_jobs_per_gpu}")
    print(f"Total concurrent threads: {len(experiment_config['available_gpus']) * args.max_jobs_per_gpu}")
    print("-" * 50)

    total_workers = len(experiment_config["available_gpus"]) * args.max_jobs_per_gpu
    
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        # Submit all jobs to the thread pool
        future_to_job = {executor.submit(run_job, job_info): job_info for job_info in all_jobs}
        
        for i, future in enumerate(as_completed(future_to_job)):
            job_info = future_to_job[future]
            flag_string = job_info[0]
            
            try:
                results_path, log_path, status = future.result()
                
                if status in ["success", "skipped"]:
                     # You can add the pickle loading check here if you want
                     print(f"({i+1}/{len(all_jobs)}) \t Job {md5(flag_string)[:6]} finished with status: {status}. Results: {results_path}")
                else:
                     print(f"({i+1}/{len(all_jobs)}) \t Job {md5(flag_string)[:6]} FAILED with status: {status}. Log: {log_path}")

            except Exception as exc:
                print(f"Job {md5(flag_string)[:6]} generated an exception: {exc}")

    print("-" * 50)
    print("All jobs have been processed.")