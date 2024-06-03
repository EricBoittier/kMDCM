# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Get calculations from NOMAD and write to JSON files.

    ...
    """

    @staticmethod
    def add_arguments(p):
        p.add_argument('uri', nargs='+', metavar='nmd://<hash>',
                       help='URIs to get')

    @staticmethod
    def run(args):
        import json

        from ase.nomad import download
        for uri in args.uri:
            calculation = download(uri)
            identifier = calculation.hash.replace('/', '.')
            fname = f'nmd.{identifier}.nomad-json'
            with open(fname, 'w') as fd:
                json.dump(calculation, fd)
            print(uri)
