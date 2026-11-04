import pytest_asyncio
from testcontainers.postgres import PostgresContainer


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vkb_dsn():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "")
