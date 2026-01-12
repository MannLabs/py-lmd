"""Integration tests for download functionality.

These tests actually download files from Zenodo and validate the content.
They require network access and may be slow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lmd._utils import _download_glyphs, _download_segmentation_example_file

if TYPE_CHECKING:
    from pathlib import Path


# TODO: Enable download of glyphs
@pytest.mark.skip(reason="Zenodo download is currently unauthorized (HTTP: 403)")
class TestDownloadGlyphsIntegration:
    """Integration tests for _download_glyphs with real downloads."""

    def test_download_glyphs_creates_valid_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that _download_glyphs downloads and extracts valid glyph files."""
        # Arrange - redirect data directory to temp path
        data_dir = tmp_path / "pylmd_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("lmd._utils._get_data_dir", lambda: data_dir)

        # Act
        result = _download_glyphs()

        # Assert
        assert result.exists()
        assert result.is_dir()

        # Check that glyph XML files were extracted
        xml_files = list(result.rglob("*.xml"))
        assert len(xml_files) > 0, "Expected glyph XML files to be extracted"

        # Verify at least some expected glyphs exist (0-9, a-i, A-I)
        glyph_names = {f.stem for f in xml_files}
        expected_glyphs = {"0", "1", "2", "a", "b", "A", "B"}
        found_glyphs = expected_glyphs & glyph_names
        assert len(found_glyphs) > 0, f"Expected some standard glyphs, found: {glyph_names}"


# TODO: Enable download of glyphs
@pytest.mark.skip(reason="Zenodo download is currently unauthorized (HTTP: 403)")
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
