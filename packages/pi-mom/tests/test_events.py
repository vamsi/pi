"""Tests for pi.mom.events."""

import json
import os
import tempfile

import pytest

from pi.mom.events import EventsWatcher, ImmediateEvent, OneShotEvent, PeriodicEvent


class TestParseEvent:
    def test_parse_immediate(self) -> None:
        content = json.dumps(
            {"type": "immediate", "channelId": "C1", "text": "hello"}
        )
        event = EventsWatcher._parse_event(content, "test.json")
        assert isinstance(event, ImmediateEvent)
        assert event.channel_id == "C1"
        assert event.text == "hello"

    def test_parse_one_shot(self) -> None:
        content = json.dumps(
            {
                "type": "one-shot",
                "channelId": "C1",
                "text": "reminder",
                "at": "2025-12-15T09:00:00+01:00",
            }
        )
        event = EventsWatcher._parse_event(content, "test.json")
        assert isinstance(event, OneShotEvent)
        assert event.at == "2025-12-15T09:00:00+01:00"

    def test_parse_periodic(self) -> None:
        content = json.dumps(
            {
                "type": "periodic",
                "channelId": "C1",
                "text": "check inbox",
                "schedule": "0 9 * * 1-5",
                "timezone": "Europe/Vienna",
            }
        )
        event = EventsWatcher._parse_event(content, "test.json")
        assert isinstance(event, PeriodicEvent)
        assert event.schedule == "0 9 * * 1-5"
        assert event.tz == "Europe/Vienna"

    def test_parse_missing_fields(self) -> None:
        content = json.dumps({"type": "immediate"})
        with pytest.raises(ValueError, match="Missing required fields"):
            EventsWatcher._parse_event(content, "test.json")

    def test_parse_unknown_type(self) -> None:
        content = json.dumps(
            {"type": "unknown", "channelId": "C1", "text": "hi"}
        )
        with pytest.raises(ValueError, match="Unknown event type"):
            EventsWatcher._parse_event(content, "test.json")

    def test_parse_one_shot_missing_at(self) -> None:
        content = json.dumps(
            {"type": "one-shot", "channelId": "C1", "text": "hi"}
        )
        with pytest.raises(ValueError, match="Missing 'at'"):
            EventsWatcher._parse_event(content, "test.json")

    def test_parse_periodic_missing_schedule(self) -> None:
        content = json.dumps(
            {
                "type": "periodic",
                "channelId": "C1",
                "text": "hi",
                "timezone": "UTC",
            }
        )
        with pytest.raises(ValueError, match="Missing 'schedule'"):
            EventsWatcher._parse_event(content, "test.json")

    def test_parse_periodic_missing_timezone(self) -> None:
        content = json.dumps(
            {
                "type": "periodic",
                "channelId": "C1",
                "text": "hi",
                "schedule": "* * * * *",
            }
        )
        with pytest.raises(ValueError, match="Missing 'timezone'"):
            EventsWatcher._parse_event(content, "test.json")

    def test_parse_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            EventsWatcher._parse_event("not json", "test.json")
