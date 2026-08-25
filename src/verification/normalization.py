"""Stable verdict labels shared by runtime code and tests."""


def normalize_verdict(verdict):
    if verdict is None:
        return "UNVERIFIED"

    normalized = str(verdict).strip().upper()
    aliases = {
        "MIXED": "PARTIALLY TRUE",
        "PARTIAL": "PARTIALLY TRUE",
        "PARTIALLY_TRUE": "PARTIALLY TRUE",
    }
    return aliases.get(normalized, normalized)
