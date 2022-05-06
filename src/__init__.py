from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

rng = np.random.default_rng()

NPImage: TypeAlias = NDArray[np.float64]
