"""Tests for agent file_id resolution (typo tolerance)."""

from letta.services.files_agents_manager import _ids_within_single_char_edit


def test_ids_within_single_char_edit_detects_one_substitution():
    good = "file-8b0f61cf-da1a-4efa-a65b-d6bc7c91660f"
    typo = "file-8b0f61cf-da1a-4efa-a65b-d6bc8c91660f"
    assert _ids_within_single_char_edit(good, typo)
    assert _ids_within_single_char_edit(typo, good)


def test_ids_within_single_char_edit_rejects_two_edits():
    a = "file-8b0f61cf-da1a-4efa-a65b-d6bc7c91660f"
    b = "file-8b0f61cf-da1a-4efa-a65b-d6bc7c91661f"
    assert not _ids_within_single_char_edit(a, b)


def test_ids_within_single_char_edit_rejects_length_mismatch():
    assert not _ids_within_single_char_edit("file-abc", "file-abcd")
