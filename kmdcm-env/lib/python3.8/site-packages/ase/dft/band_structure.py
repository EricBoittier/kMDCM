import warnings

from numpy import VisibleDeprecationWarning

from ase.spectrum.band_structure import *  # noqa: F401,F403

warnings.warn("ase.dft.band_structure has been moved to "
              "ase.spectrum.band_structure. Please update your "
              "scripts; this alias will be removed in a future "
              "version of ASE.",
              VisibleDeprecationWarning)
