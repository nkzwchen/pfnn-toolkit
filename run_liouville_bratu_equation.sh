#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#!/bin/bash
module load anaconda/2020.11
module load openmpi/4.1.1
module load cuda/10.1
module load nccl/2.9.6-1_cuda10.1
module load cudnn/7.6.5.32_cuda10.1
source activate mindspore
mpirun -n $1 python liouville_bratu_equation.py --g_epochs 6000 --f_epochs 6000 --g_lr 0.01 --f_lr 0.01 --parallel_mode DATA_PARALLEL
