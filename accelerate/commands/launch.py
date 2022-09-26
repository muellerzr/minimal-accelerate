#!/usr/bin/env python

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import logging
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Dict, List

import torch

import psutil
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.state import get_int_from_env
from accelerate.utils import (
    ComputeEnvironment,
    DistributedType,
    PrecisionType,
    PrepareForLaunch,
    _filter_args,
    is_torch_version,
    patch_environment,
)


if is_torch_version(">=", "1.9.0"):
    import torch.distributed.run as distrib_run

logger = logging.getLogger(__name__)


def launch_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("launch")
    else:
        parser = argparse.ArgumentParser("Accelerate launch command")

    parser.add_argument(
        "--config_file", default=None, help="The config file to use for the default values in the launching script."
    )
    parser.add_argument(
        "--multi_gpu",
        default=False,
        action="store_true",
        help="Whether or not this should launch a distributed GPU training.",
    )
    parser.add_argument(
        "--use_mps_device",
        default=False,
        action="store_true",
        help="Whether or not this should use MPS-enabled GPU device on MacOS machines.",
    )
    parser.add_argument(
        "--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
    )

    parser.add_argument(
        "--fp16", default=False, action="store_true", help="Whether or not to use mixed precision training."
    )
    parser.add_argument(
        "--cpu", default=False, action="store_true", help="Whether or not to force the training on the CPU."
    )
    parser.add_argument(
        "--num_processes", type=int, default=None, help="The total number of processes to be launched in parallel."
    )
    parser.add_argument(
        "--num_machines", type=int, default=None, help="The total number of machines used in this training."
    )
    parser.add_argument(
        "--machine_rank", type=int, default=None, help="The rank of the machine on which this script is launched."
    )
    parser.add_argument("--main_process_ip", type=str, default=None, help="The IP address of the machine of rank 0.")
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="The port to use to communicate with the machine of rank 0.",
    )
    # Rendezvous related arguments
    parser.add_argument(
        "--rdzv_conf",
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--main_training_function",
        type=str,
        default=None,
        help="The name of the main function to be executed in your script (only for TPU training).",
    )
    parser.add_argument(
        "--downcast_bf16",
        action="store_true",
        help="Whether when using bf16 precision on TPUs if both float and double tensors are cast to bfloat16 or if double tensors remain as float32",
    )
    parser.add_argument(
        "-m",
        "--module",
        action="store_true",
        help="Change each process to interpret the launch script as a Python module, executing with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        action="store_true",
        help="Skip prepending the training script with 'python' - just execute it directly. Useful when the script is not a Python script.",
    )
    parser.add_argument(
        "--num_cpu_threads_per_process",
        type=int,
        default=None,
        help="The number of CPU threads per process. Can be tuned for optimal performance.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print out the torch.distributed stack trace when something fails.",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the script to be launched in parallel, followed by all the arguments for the training "
            "script."
        ),
    )

    # Other arguments of the training scripts
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER, help="Arguments of the training script.")

    if subparsers is not None:
        parser.set_defaults(func=launch_command)
    return parser


def simple_launcher(args):
    cmd = []
    if args.no_python and args.module:
        raise ValueError("--module and --no_python cannot be used together")
    if not args.no_python:
        cmd.append(sys.executable)
        if args.module:
            cmd.append("-m")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["USE_CPU"] = str(args.cpu or args.use_cpu)
    current_env["USE_MPS_DEVICE"] = str(args.use_mps_device)
    if args.use_mps_device:
        current_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if args.num_machines > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip
        current_env["MASTER_PORT"] = str(args.main_process_port)
    elif args.num_processes > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip if args.main_process_ip is not None else "127.0.0.1"
        current_env["MASTER_PORT"] = str(args.main_process_port) if args.main_process_port is not None else "29500"

    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    current_env["MIXED_PRECISION"] = str(mixed_precision)
    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def multi_gpu_launcher(args):
    num_processes = getattr(args, "num_processes")
    num_machines = getattr(args, "num_machines")
    main_process_ip = getattr(args, "main_process_ip")
    main_process_port = getattr(args, "main_process_port")
    if num_machines > 1:
        setattr(args, "nproc_per_node", str(num_processes // num_machines))
        setattr(args, "nnodes", str(num_machines))
        setattr(args, "node_rank", int(args.machine_rank))
        if getattr(args, "same_network"):
            setattr(args, "master_addr", str(main_process_ip))
            setattr(args, "master_port", str(main_process_port))
        else:
            setattr(args, "rdzv_endpoint", f"{main_process_ip}:{main_process_port}")
    else:
        setattr(args, "nproc_per_node", str(num_processes))
        if main_process_port is not None:
            setattr(args, "master_port", str(main_process_port))

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        setattr(args, "module", True)
    elif args.no_python:
        setattr(args, "no_python", True)

    current_env = os.environ.copy()
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.")

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    current_env["MIXED_PRECISION"] = str(mixed_precision)
    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    if is_torch_version("<", "1.9.0"):
        raise NotImplementedError("Multi-node training requires pytorch>=1.9.0")

    args = _filter_args(args)
    with patch_environment(**current_env):
        distrib_run.run(args)

def tpu_launcher(args):
    import torch_xla.distributed.xla_multiprocessing as xmp

    current_env = {}

    if args.no_python:
        raise ValueError("--no_python cannot be used with TPU launcher")

    if args.mixed_precision == "bf16":
        if args.downcast_bf16:
            current_env["XLA_USE_BF16"] = "0"
            current_env["XLA_DOWNCAST_BF16"] = "1"
        else:
            current_env["XLA_USE_BF16"] = "1"
            current_env["XLA_DOWNCAST_BF16"] = "0"

    if args.module:
        mod_name = args.training_script
    else:
        # Import training_script as a module
        script_path = Path(args.training_script)
        sys.path.append(str(script_path.parent.resolve()))
        mod_name = script_path.stem

    mod = importlib.import_module(mod_name)
    if not hasattr(mod, args.main_training_function):
        raise ValueError(
            f"Your training script should have a function named {args.main_training_function}, or you should pass a "
            "different value to `--main_training_function`."
        )

    # Patch sys.argv
    sys.argv = [mod.__file__] + args.training_script_args

    main_function = getattr(mod, args.main_training_function)
    with patch_environment(**current_env):
        xmp.spawn(PrepareForLaunch(main_function), args=(), nprocs=args.num_processes)



def launch_command(args):
    # Sanity checks
    if sum([args.multi_gpu, args.tpu]) > 1:
        raise ValueError("You can only pick one between `--multi_gpu`, `--tpu`.")

    defaults = None
    warned = []
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file) and not args.cpu:
        defaults = load_config_from_file(args.config_file)
        if (
            not args.multi_gpu
            and not args.tpu
            and not args.use_mps_device
        ):
            args.multi_gpu = defaults.distributed_type == DistributedType.MULTI_GPU
            args.tpu = defaults.distributed_type == DistributedType.TPU
            args.use_mps_device = defaults.distributed_type == DistributedType.MPS

        if defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            # Update args with the defaults
            for name, attr in defaults.__dict__.items():
                # Those args are handled separately
                if (
                    name not in ["compute_environment", "fp16", "mixed_precision", "distributed_type"]
                    and getattr(args, name, None) is None
                ):
                    setattr(args, name, attr)
        if not args.mixed_precision:
            if args.fp16:
                args.mixed_precision = "fp16"
            else:
                args.mixed_precision = defaults.mixed_precision
    else:
        if args.num_processes is None:
            warned.append("\t`--num_processes` was set to a value of `1`")
            args.num_processes = 1
        if args.num_machines is None:
            warned.append("\t`--num_machines` was set to a value of `1`")
            args.num_machines = 1
        if args.mixed_precision is None:
            warned.append("\t`--mixed_precision` was set to a value of `'no'`")
            args.mixed_precision = "no"
        if not hasattr(args, "use_cpu"):
            args.use_cpu = args.cpu
    if args.multi_gpu and args.num_processes == 1:
        args.num_processes = torch.cuda.device_count()
        if not any("--num_processes" in warn for warn in warned):
            warned.append(f"\t`--num_processes` was set to `{args.num_processes}`")
        else:
            for i, warn in enumerate(warned):
                if "--num_processes" in warn:
                    warned[i] = warn.replace("`1`", f"`{args.num_processes}`")

    if args.num_cpu_threads_per_process is None:
        local_size = get_int_from_env(
            ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
        )
        args.num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
        if args.num_cpu_threads_per_process == 0:
            args.num_cpu_threads_per_process = 1
        warned.append(
            f"\t`--num_cpu_threads_per_process` was set to `{args.num_cpu_threads_per_process}` to improve out-of-box performance"
        )

    if any(warned):
        message = "The following values were not passed to `accelerate launch` and had defaults used instead:\n"
        message += "\n".join(warned)
        message += (
            "\nTo avoid this warning pass in values for each of the problematic parameters or run `accelerate config`."
        )
        logger.warn(message)

    # Use the proper launcher
    if args.multi_gpu and not args.cpu:
        multi_gpu_launcher(args)
    elif args.tpu and not args.cpu:
        tpu_launcher(args)
    else:
        simple_launcher(args)


def main():
    parser = launch_command_parser()
    args = parser.parse_args()
    launch_command(args)


if __name__ == "__main__":
    main()
