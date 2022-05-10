from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

rng: np.random.Generator = np.random.default_rng()
NpImageDType = np.float32
NpImage: TypeAlias = NDArray[NpImageDType]
