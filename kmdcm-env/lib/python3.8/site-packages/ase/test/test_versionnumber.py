import os
from importlib.metadata import PackageNotFoundError, version

import ase


def test_versionnumber():
    # Check that the version number reported by importlib matches
    # what is hardcoded in ase/__init__.py
    try:
        version_seen_by_python = version('ase')
    except PackageNotFoundError:
        pass
    else:
        assert ase.__version__ == version_seen_by_python, \
            f'ASE is version {ase.__version__} but python think it is ' + \
            f'{version_seen_by_python} - perhaps rerun "pip install -e"'

    # Check that version numbers in ase/__init__.py and pyproject.toml match
    asedir = os.path.dirname(ase.__file__)
    # Possibly follow a symlink
    asedir = os.path.realpath(asedir)
    # Go one level up
    asedir = os.path.dirname(asedir)
    # Find config file
    configfile = os.path.join(asedir, 'pyproject.toml')
    if os.path.isfile(configfile):
        # Parse it without requiring new dependencies.
        with open(configfile, "rt") as toml:
            projectsection = False
            tomlversion = 'version not found in pyproject.toml'
            for line in toml:
                if '#' in line:
                    line = line.split('#')[0]
                line = line.strip()
                if line and line[0] == '[' and line[-1] == ']':
                    projectsection = line == '[project]'
                if projectsection:
                    words = line.split('=')
                    if words[0].strip() == 'version':
                        tomlversion = words[1].strip()[1:-1]
                        print(f'Found version in {configfile}: {tomlversion}')
        assert ase.__version__ == tomlversion, \
            'Version number in ase/__init__.py and pyproject.toml do not match'
