import os
import numpy as np

from torch import nn

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import DigitalRankUpdateRPUConfig
from aihwkit.simulator.configs.devices import (
    MixedPrecisionCompound,
    SoftBoundsDevice)
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import (ConstantStepDevice, TransferCompound, SoftBoundsDevice)


rpu_config = DigitalRankUpdateRPUConfig(
    device=MixedPrecisionCompound(
        device=SoftBoundsDevice(),
    )
)


to_save = np.ones((3,4))

tag = 1


os.makedirs('tests', exist_ok=True)

with open("tests/foo"+str(tag)+".txt", "w") as f:
    f.write(str(rpu_config))

#np.savetxt(os.path.join('tests', 'testtest'+str(tag)+'.txt'), print(rpu_config))

INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10 
RPU_CONFIG = SingleRPUConfig(device=ConstantStepDevice())


model = AnalogSequential(
    AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True, rpu_config=RPU_CONFIG),
    nn.Sigmoid(),
    AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True, rpu_config=RPU_CONFIG),
    nn.Sigmoid(),
    AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True, rpu_config=RPU_CONFIG),
    nn.LogSoftmax(dim=1)
)

print(model[0].analog_tile.tile)