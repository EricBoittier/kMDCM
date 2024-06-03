import io

from ase.io import read

header = """
  ___ ___ ___ _ _ _
 |   |   |_  | | | |
 | | | | | . | | | |
 |__ |  _|___|_____|  21.1.0
 |___|_|
"""

atoms = """
Reference energy: -26313.685229

Positions:
   0 Al     0.000000    0.000000    0.000000    ( 0.0000,  0.0000,  0.0000)

Unit cell:
           periodic     x           y           z      points  spacing
  1. axis:    yes    4.050000    0.000000    0.000000    21     0.1929
  2. axis:    yes    0.000000    4.050000    0.000000    21     0.1929
  3. axis:    yes    0.000000    0.000000    4.050000    21     0.1929

Energy contributions relative to reference atoms: (reference = -26313.685229)

Kinetic:        +23.028630
Potential:       -8.578488
External:        +0.000000
XC:             -24.279425
Entropy (-ST):   -0.381921
Local:           -0.018721
Extra
stuff:          117.420000
--------------------------
Free energy:    -10.229926
Extrapolated:   -10.038965
"""

forces = """
Forces in eV/Ang:
  0 Al    0.00000    0.00000   -0.00000
"""

stress = """
Stress tensor:
     0.000000     0.000000     0.000000
     0.000000     0.000000     0.000000
     0.000000     0.000000     0.000000"""

# Three configurations.  Only 1. and 3. has forces.
text = header + atoms + forces + atoms + atoms + forces + stress


def test_gpaw_output():
    """Regression test for #896.

    "ase.io does not read all configurations from gpaw-out file"

    """
    fd = io.StringIO(text)
    configs = read(fd, index=':', format='gpaw-out')
    assert len(configs) == 3
