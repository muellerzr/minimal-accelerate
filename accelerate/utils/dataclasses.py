# Copyright 2022 The HuggingFace Team. All rights reserved.
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

"""
General namespace and dataclass related classes
"""

import copy
import enum
import functools
import os
import typing
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Iterable, Optional

import torch

class KwargsHandler:
    """
    Internal mixin that implements a `to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how your model is wrapped in a
    `torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
    [wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
    information on each argument.

    <Tip warning={true}>

    `gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.

    `static_graph` is only available in PyTorch 1.11.0 and later versions.

    </Tip>"""

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `GradScaler` is only available in PyTorch 1.5.0 and later versions.

    </Tip>"""

    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True


@dataclass
class InitProcessGroupKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the distributed processes. Please refer
    to the documentation of this
    [method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
    information on each argument.
    """

    init_method: Optional[str] = None
    timeout: timedelta = timedelta(seconds=1800)


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    TPU = "TPU"
    MPS = "MPS"


class SageMakerDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **DATA_PARALLEL** -- using sagemaker distributed data parallelism.
        - **MODEL_PARALLEL** -- using sagemaker distributed model parallelism.
    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    DATA_PARALLEL = "DATA_PARALLEL"
    MODEL_PARALLEL = "MODEL_PARALLEL"


class ComputeEnvironment(str, enum.Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """

    # Subclassing str as well as Enum allows the `ComputeEnvironment` to be JSON-serializable out of the box.
    LOCAL_MACHINE = "LOCAL_MACHINE"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"


class EnumWithContains(enum.EnumMeta):
    "A metaclass that adds the ability to check if `self` contains an item with the `in` operator"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    "An enum class that can get the value of an item with `str(Enum.key)`"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        "Method to list all the possible items in `cls`"
        return list(map(lambda item: str(item), cls))


class LoggerType(BaseEnum):
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
    """

    ALL = "all"
    AIM = "aim"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"


class PrecisionType(BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class RNGType(BaseEnum):
    TORCH = "torch"
    CUDA = "cuda"
    XLA = "xla"
    GENERATOR = "generator"


# data classes


@dataclass
class TensorInformation:
    shape: torch.Size
    dtype: torch.dtype