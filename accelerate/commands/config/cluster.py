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

from ...utils import ComputeEnvironment, DistributedType, is_deepspeed_available, is_transformers_available
from .config_args import ClusterConfig
from .config_utils import _ask_field, _convert_distributed_mode, _convert_yes_no_to_bool


def get_cluster_input():
    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): ",
        _convert_distributed_mode,
        error_message="Please enter 0, 1, 2, 3 or 4.",
    )

    machine_rank = 0
    num_machines = 1
    main_process_ip = None
    main_process_port = None
    rdzv_backend = "static"
    same_network = True
    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.MULTI_CPU]:
        num_machines = _ask_field(
            "How many different machines will you use (use more than 1 for multi-node training)? [1]: ",
            lambda x: int(x),
            default=1,
        )
        if num_machines > 1:
            machine_rank = _ask_field(
                "What is the rank of this machine (from 0 to the number of machines - 1 )? [0]: ",
                lambda x: int(x),
                default=0,
            )
            main_process_ip = _ask_field(
                "What is the IP address of the machine that will host the main process? ",
            )
            main_process_port = _ask_field(
                "What is the port you will use to communicate with the main process? ",
                lambda x: int(x),
            )
            same_network = _ask_field(
                "Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: ",
                _convert_yes_no_to_bool,
                default=True,
                error_message="Please enter yes or no.",
            )
            if not same_network:
                rdzv_backend = _ask_field(
                    "What rendezvous backend will you use? ('static', 'c10d', ...): ", default="static"
                )

    if distributed_type == DistributedType.NO:
        use_cpu = _ask_field(
            "Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
    elif distributed_type == DistributedType.MULTI_CPU:
        use_cpu = True
    else:
        use_cpu = False

    if distributed_type == DistributedType.TPU:
        main_training_function = _ask_field(
            "What is the name of the function in your script that should be launched in all parallel scripts? [main]: ",
            default="main",
        )
    else:
        main_training_function = "main"

    if distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_GPU, DistributedType.TPU]:
        machine_type = str(distributed_type).split(".")[1].replace("MULTI_", "")
        if machine_type == "TPU":
            machine_type += " cores"
        else:
            machine_type += "(s)"
        num_processes = _ask_field(
            f"How many {machine_type} should be used for distributed training? [1]:",
            lambda x: int(x),
            default=1,
            error_message="Please enter an integer.",
        )
    else:
        num_processes = 1

    if distributed_type != DistributedType.TPU:
        mixed_precision = _ask_field(
            "Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: ",
            lambda x: str(x).lower(),
            default="no",
        )
    else:
        mixed_precision = "no"

    downcast_bf16 = "no"
    if distributed_type == DistributedType.TPU and mixed_precision == "bf16":
        downcast_bf16 = _ask_field(
            "Should `torch.float` be cast as `bfloat16` and `torch.double` remain `float32` on TPUs?", default="no"
        )

    return ClusterConfig(
        compute_environment=ComputeEnvironment.LOCAL_MACHINE,
        distributed_type=distributed_type,
        num_processes=num_processes,
        mixed_precision=mixed_precision,
        downcast_bf16=downcast_bf16,
        machine_rank=machine_rank,
        num_machines=num_machines,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
        main_training_function=main_training_function,
        use_cpu=use_cpu,
        rdzv_backend=rdzv_backend,
        same_network=same_network,
    )
