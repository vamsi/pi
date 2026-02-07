"""Tests for pi.mom.store."""

import json
import os
import tempfile

import pytest

from pi.mom.store import Attachment, ChannelStore, LoggedMessage


@pytest.fixture
def tmpdir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestChannelStore:
    def test_get_channel_dir_creates(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        d = store.get_channel_dir("C123")
        assert os.path.isdir(d)
        assert d.endswith("C123")

    def test_generate_local_filename(self) -> None:
        name = ChannelStore.generate_local_filename("my file (1).png", "1732531234.567890")
        assert name.startswith("1732531234567_")
        assert "my_file__1_.png" in name

    def test_generate_local_filename_sanitizes(self) -> None:
        name = ChannelStore.generate_local_filename("bad/chars:here!", "1000000000.000000")
        assert "/" not in name
        assert ":" not in name

    @pytest.mark.asyncio
    async def test_log_message(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        msg = LoggedMessage(
            date="2025-01-01T00:00:00Z",
            ts="1234567890.123456",
            user="U123",
            text="hello",
            attachments=[],
            is_bot=False,
            user_name="mario",
        )
        result = await store.log_message("C456", msg)
        assert result is True

        log_path = os.path.join(tmpdir, "C456", "log.jsonl")
        assert os.path.exists(log_path)

        with open(log_path) as f:
            data = json.loads(f.readline())
        assert data["user"] == "U123"
        assert data["text"] == "hello"
        assert data["userName"] == "mario"

    @pytest.mark.asyncio
    async def test_log_message_deduplicates(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        msg = LoggedMessage(
            date="2025-01-01T00:00:00Z",
            ts="111.222",
            user="U1",
            text="hi",
            attachments=[],
            is_bot=False,
        )
        assert await store.log_message("C1", msg) is True
        assert await store.log_message("C1", msg) is False  # duplicate

    def test_get_last_timestamp(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        assert store.get_last_timestamp("C1") is None

        # Write some log entries
        ch_dir = store.get_channel_dir("C1")
        log_path = os.path.join(ch_dir, "log.jsonl")
        with open(log_path, "w") as f:
            f.write(json.dumps({"ts": "100.0"}) + "\n")
            f.write(json.dumps({"ts": "200.0"}) + "\n")

        assert store.get_last_timestamp("C1") == "200.0"

    @pytest.mark.asyncio
    async def test_log_bot_response(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        await store.log_bot_response("C1", "bot says hi", "999.0")

        log_path = os.path.join(tmpdir, "C1", "log.jsonl")
        with open(log_path) as f:
            data = json.loads(f.readline())
        assert data["user"] == "bot"
        assert data["isBot"] is True
        assert data["text"] == "bot says hi"

    def test_process_attachments(self, tmpdir: str) -> None:
        store = ChannelStore(tmpdir, "xoxb-fake")
        files = [
            {"name": "test.png", "url_private_download": "https://example.com/test.png"},
            {"name": "doc.pdf", "url_private": "https://example.com/doc.pdf"},
            {"url_private": "https://example.com/noname"},  # no name - should skip
        ]
        atts = store.process_attachments("C1", files, "1000.000")
        assert len(atts) == 2
        assert atts[0].original == "test.png"
        assert "C1/attachments/" in atts[0].local
        assert atts[1].original == "doc.pdf"
