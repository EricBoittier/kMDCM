import os

import pytest

from ase.db import connect


@pytest.fixture(scope='session')
def mysql_port():
    return int(os.environ.get('MYSQL_TCP_PORT', 3306))


@pytest.fixture()
def get_db_name(mysql_port):
    """ Fixture that returns a function to get the test db name
    for the different supported db types.

    Args:
        dbtype (str): Type of database. Currently only 5 types supported:
            postgresql, mysql, mariadb, json, and db (sqlite3)
        clean_db (bool): Whether to clean all entries from the db. Useful
            for reusing the database across multiple tests. Defaults to True.
    """
    def _func(dbtype, clean_db=True):
        name = None

        if dbtype == 'postgresql':
            pytest.importorskip('psycopg2')
            if os.environ.get('POSTGRES_DB'):  # gitlab-ci
                name = 'postgresql://ase:ase@postgres:5432/testase'
            else:
                name = os.environ.get('ASE_TEST_POSTGRES_URL')
        elif dbtype == 'mysql':
            pytest.importorskip('pymysql')
            if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
                # Note: testing of non-standard port by changing from default
                # of 3306 to 3307
                name = f'mysql://root:ase@mysql:{mysql_port}/testase_mysql'
            else:
                name = os.environ.get('MYSQL_DB_URL')
        elif dbtype == 'mariadb':
            pytest.importorskip('pymysql')
            if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
                # Note: testing of non-standard port by changing from default
                # of 3306 to 3307
                name = f'mariadb://root:ase@mariadb:{mysql_port}/testase_mysql'
            else:
                name = os.environ.get('MYSQL_DB_URL')
        elif dbtype == 'json':
            name = 'testase.json'
        elif dbtype == 'db':
            name = 'testase.db'
        else:
            raise ValueError(f'Bad db type: {dbtype}')

        if name is None:
            pytest.skip('Test requires environment variables')

        if clean_db:
            if dbtype in ["postgresql", "mysql", "mariadb"]:
                c = connect(name)
                c.delete([row.id for row in c.select()])

        return name

    return _func
