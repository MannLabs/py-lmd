"""Integration tests for download functionality.

These tests actually download files from Zenodo and validate the content.
They require network access and may be slow.

NOTE: While it is unusual to call third-party services in tests (typically you
would mock requests), these integration tests serve as end-to-end validation
that the download URLs and extraction logic work correctly with the real
upstream data. They have proven valuable in catching real issues (e.g., server
header requirements, URL changes). The unit tests in test_utils.py mock the
network layer for fast, isolated testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lmd._utils import _download_glyphs, _download_segmentation_example_file

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestDownloadGlyphsIntegration:
    """Integration tests for _download_glyphs with real downloads."""

    def test_download_glyphs_creates_valid_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that _download_glyphs downloads and extracts valid glyph files."""
        # Arrange - redirect data directory to temp path
        data_dir = tmp_path / "pylmd_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("lmd._utils._get_data_dir", lambda: data_dir)

        result = _download_glyphs()

        assert result.exists()
        assert result.is_dir()

        # Check that glyph SVG files were extracted
        svg_files = list(result.rglob("*.svg"))
        assert len(svg_files) > 0, "Expected glyph SVG files to be extracted"

        # Verify at least some expected glyphs exist (0-9, a-i, A-I)
        glyph_names = {f.stem for f in svg_files}
        expected_glyphs = {"0", "1", "2", "a", "b", "A", "B"}
        found_glyphs = expected_glyphs & glyph_names
        assert len(found_glyphs) > 0, f"Expected some standard glyphs, found: {glyph_names}"


class TestDownloadSegmentationIntegration:
    """Integration tests for _download_segmentation_example_file with real downloads."""

    def test_download_segmentation_creates_valid_tiff(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that _download_segmentation_example_file downloads a valid TIFF."""
        from PIL import Image

        # Arrange
        data_dir = tmp_path / "pylmd_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("lmd._utils._get_data_dir", lambda: data_dir)

        # Act
        result = _download_segmentation_example_file()

        # Assert
        assert result.name == "segmentation_mask.tif"
        assert result.parent.exists()

        # Verify it's a valid TIFF that can be opened
        img = Image.open(result)
        assert img.size[0] > 0
        assert img.size[1] > 0
