import pytest
from pi.web.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    """Create an in-memory database for testing."""
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()
