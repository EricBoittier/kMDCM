"""Test Pourbaix diagram."""
import numpy as np
import pytest

from ase.phasediagram import Pourbaix, solvated


def test_pourbaix():
    """Test ZnO system from docs."""
    refs = solvated('Zn')
    print(refs)
    refs += [('Zn', 0.0), ('ZnO', -3.323), ('ZnO2(aq)', -2.921)]
    pb = Pourbaix(refs, formula='ZnO')

    _, e = pb.decompose(-1.0, 7.0)
    assert e == pytest.approx(-3.625, abs=0.001)

    U = np.linspace(-2, 2, 3)
    pH = np.linspace(6, 16, 11)
    d, names, text = pb.diagram(U, pH, plot=False)
    print(d, names, text)
    assert d.shape == (3, 11)
    assert np.ptp(d) == 6
    assert names == ['Zn', 'ZnO2(aq)', 'Zn++(aq)', 'HZnO2-(aq)',
                     'ZnOH+(aq)', 'ZnO', 'ZnO2--(aq)']
