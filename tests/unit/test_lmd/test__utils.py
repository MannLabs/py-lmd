from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from lmd._utils import (
    _download,
    _download_glyphs,
    _download_segmentation_example_file,
    _get_data_dir,
)


class TestGetDataDir:
    """Tests for _get_data_dir function."""

    def test_returns_path(self) -> None:
        """Test that _get_data_dir returns an absolute Path object that ends with .pylmd."""
        result = _get_data_dir()

        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result.name == ".pylmd"


class TestDownload:
    """Tests for _download function."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Fixture providing a temporary directory for downloads."""
        return tmp_path / "downloads"

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Fixture providing a mocked requests response."""
        mock = MagicMock()
        mock.headers = {"content-length": "100"}
        mock.iter_content.return_value = [b"test_content"]
        return mock

    def test_download_creates_output_directory(self, temp_dir: Path, mock_response: MagicMock) -> None:
        """Test that _download creates the output directory if it doesn't exist."""
        assert not temp_dir.exists()

        with patch("lmd._utils.requests.get", return_value=mock_response):
            _download(
                url="https://example.com/file.txt",
                output_path=str(temp_dir),
                output_file_name="test_file.txt",
            )

        assert temp_dir.exists()

    def test_download_creates_file(self, temp_dir: Path, mock_response: MagicMock) -> None:
        """Test that _download creates the downloaded file with the expected content."""
        output_file = "test_file.txt"

        with patch("lmd._utils.requests.get", return_value=mock_response):
            _download(
                url="https://example.com/file.txt",
                output_path=str(temp_dir),
                output_file_name=output_file,
            )

        # File exists
        assert (temp_dir / output_file).exists()

        # Content is correct
        with open(temp_dir / output_file, "rb") as f:
            content = f.read()
        assert content == b"test_content"

    def test_download_skips_existing_file_without_overwrite(
        self, temp_dir: Path, mock_response: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that _download skips download when file exists and overwrite=False."""
        # Arrange
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_file = "existing_file.txt"
        existing_file = temp_dir / output_file
        existing_file.write_text("existing_content")

        with patch("lmd._utils.requests.get", return_value=mock_response) as mock_get:
            _download(
                url="https://example.com/file.txt",
                output_path=str(temp_dir),
                output_file_name=output_file,
                overwrite=False,
            )

        mock_get.assert_not_called()
        assert existing_file.read_text() == "existing_content"
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_download_overwrites_existing_file_with_overwrite(
        self, temp_dir: Path, mock_response: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that _download overwrites file when overwrite=True."""
        # Arrange
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_file = "existing_file.txt"
        existing_file = temp_dir / output_file
        existing_file.write_text("existing_content")

        # Act
        with patch("lmd._utils.requests.get", return_value=mock_response):
            _download(
                url="https://example.com/file.txt",
                output_path=str(temp_dir),
                output_file_name=output_file,
                overwrite=True,
            )

        # Assert
        with open(existing_file, "rb") as f:
            content = f.read()
        assert content == b"test_content"
        captured = capsys.readouterr()
        assert "Overwriting" in captured.out

    def test_download_generates_random_filename_when_none(self, temp_dir: Path, mock_response: MagicMock) -> None:
        """Test that _download generates a random filename when output_file_name is None."""
        # Arrange & Act
        with patch("lmd._utils.requests.get", return_value=mock_response):
            _download(
                url="https://example.com/file.txt",
                output_path=str(temp_dir),
                output_file_name=None,
            )

        # Assert
        files = list(temp_dir.iterdir())
        # Filter out lock files
        files = [f for f in files if not f.name.endswith(".lock")]
        assert len(files) == 1
        assert files[0].name.startswith("pylmd_tmp_")

    def test_download_extracts_zip_archive(self, temp_dir: Path, mock_response: MagicMock) -> None:
        """Test that _download extracts zip archives when archive_format is specified."""
        # Arrange
        output_file = "archive.zip"

        # Act
        with (
            patch("lmd._utils.requests.get", return_value=mock_response),
            patch("lmd._utils.shutil.unpack_archive") as mock_unpack,
            patch("lmd._utils.os.remove") as mock_remove,
        ):
            _download(
                url="https://example.com/archive.zip",
                output_path=str(temp_dir),
                output_file_name=output_file,
                archive_format="zip",
            )

        # Assert
        mock_unpack.assert_called_once()
        mock_remove.assert_called_once()

    @pytest.mark.parametrize(
        "archive_format",
        ["zip", "tar", "tar.gz", "tgz"],
        ids=["zip", "tar", "tar.gz", "tgz"],
    )
    def test_download_supports_archive_formats(
        self, temp_dir: Path, mock_response: MagicMock, archive_format: Literal["zip", "tar", "tar.gz", "tgz"]
    ) -> None:
        """Test that _download supports various archive formats."""
        # Arrange
        output_file = f"archive.{archive_format}"

        # Act
        with (
            patch("lmd._utils.requests.get", return_value=mock_response),
            patch("lmd._utils.shutil.unpack_archive") as mock_unpack,
            patch("lmd._utils.os.remove"),
        ):
            _download(
                url=f"https://example.com/{output_file}",
                output_path=str(temp_dir),
                output_file_name=output_file,
                archive_format=archive_format,
            )

        # Assert
        call_args = mock_unpack.call_args
        assert call_args[1]["format"] == archive_format

    def test_download_handles_path_with_trailing_slash(self, temp_dir: Path, mock_response: MagicMock) -> None:
        """Test that _download handles output_path with trailing slash correctly."""
        # Arrange
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_file = "test_file.txt"
        path_with_slash = str(temp_dir) + "/"

        # Act
        with patch("lmd._utils.requests.get", return_value=mock_response):
            _download(
                url="https://example.com/file.txt",
                output_path=path_with_slash,
                output_file_name=output_file,
            )

        # Assert
        assert (temp_dir / output_file).exists()


class TestDownloadGlyphs:
    """Tests for _download_glyphs function."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path: Path) -> Path:
        """Fixture providing a mock data directory."""
        data_dir = tmp_path / ".pylmd"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def test_returns_path(self, mock_data_dir: Path) -> None:
        """Test that _download_glyphs returns a Path object."""
        glyphs_path = mock_data_dir / "glyphs"
        glyphs_path.mkdir(parents=True, exist_ok=True)
        # Create a dummy file to make directory non-empty
        (glyphs_path / "dummy.txt").touch()

        with patch("lmd._utils._get_data_dir", return_value=mock_data_dir):
            result = _download_glyphs()

        assert isinstance(result, Path)

    def test_returns_glyphs_subdirectory(self, mock_data_dir: Path) -> None:
        """Test that _download_glyphs returns path to glyphs subdirectory."""
        glyphs_path = mock_data_dir / "glyphs"
        glyphs_path.mkdir(parents=True, exist_ok=True)
        (glyphs_path / "dummy.txt").touch()

        with patch("lmd._utils._get_data_dir", return_value=mock_data_dir):
            result = _download_glyphs()

        assert result.name == "glyphs"

    def test_downloads_when_directory_does_not_exist(self, mock_data_dir: Path) -> None:
        """Test that _download_glyphs downloads when glyphs directory doesn't exist."""
        glyphs_path = mock_data_dir / "glyphs"
        assert not glyphs_path.exists()

        with (
            patch("lmd._utils._get_data_dir", return_value=mock_data_dir),
            patch("lmd._utils._download") as mock_download,
        ):
            _download_glyphs()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["archive_format"] == "zip"

    def test_downloads_when_directory_is_empty(self, mock_data_dir: Path) -> None:
        """Test that _download_glyphs downloads when glyphs directory exists but is empty."""
        glyphs_path = mock_data_dir / "glyphs"
        glyphs_path.mkdir(parents=True, exist_ok=True)
        assert glyphs_path.exists()
        assert not any(glyphs_path.iterdir())

        with (
            patch("lmd._utils._get_data_dir", return_value=mock_data_dir),
            patch("lmd._utils._download") as mock_download,
        ):
            _download_glyphs()

        mock_download.assert_called_once()

    def test_skips_download_when_directory_has_files(self, mock_data_dir: Path) -> None:
        """Test that _download_glyphs skips download when glyphs directory has files."""
        glyphs_path = mock_data_dir / "glyphs"
        glyphs_path.mkdir(parents=True, exist_ok=True)
        (glyphs_path / "glyph_a.xml").touch()

        with (
            patch("lmd._utils._get_data_dir", return_value=mock_data_dir),
            patch("lmd._utils._download") as mock_download,
        ):
            _download_glyphs()

        mock_download.assert_not_called()


class TestDownloadSegmentationExampleFile:
    """Tests for _download_segmentation_example_file function."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path: Path) -> Path:
        """Fixture providing a mock data directory."""
        data_dir = tmp_path / ".pylmd"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def test_returns_path(self, mock_data_dir: Path) -> None:
        """Test that _download_segmentation_example_file returns a Path object."""
        seg_path = mock_data_dir / "segmentation_cytosol_example"
        seg_path.mkdir(parents=True, exist_ok=True)

        with patch("lmd._utils._get_data_dir", return_value=mock_data_dir):
            result = _download_segmentation_example_file()

        assert isinstance(result, Path)

    def test_returns_tif_file_path(self, mock_data_dir: Path) -> None:
        """Test that _download_segmentation_example_file returns path to tif file."""
        seg_path = mock_data_dir / "segmentation_cytosol_example"
        seg_path.mkdir(parents=True, exist_ok=True)

        with patch("lmd._utils._get_data_dir", return_value=mock_data_dir):
            result = _download_segmentation_example_file()

        assert result.name == "segmentation_mask.tif"

    def test_downloads_when_directory_does_not_exist(self, mock_data_dir: Path) -> None:
        """Test that function downloads when segmentation directory doesn't exist."""
        seg_path = mock_data_dir / "segmentation_cytosol_example"
        assert not seg_path.exists()

        with (
            patch("lmd._utils._get_data_dir", return_value=mock_data_dir),
            patch("lmd._utils._download") as mock_download,
        ):
            _download_segmentation_example_file()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["output_file_name"] == "segmentation_mask.tif"
        assert call_kwargs["archive_format"] is None

    def test_skips_download_when_directory_exists(self, mock_data_dir: Path) -> None:
        """Test that function skips download when segmentation directory exists."""
        seg_path = mock_data_dir / "segmentation_cytosol_example"
        seg_path.mkdir(parents=True, exist_ok=True)

        with (
            patch("lmd._utils._get_data_dir", return_value=mock_data_dir),
            patch("lmd._utils._download") as mock_download,
        ):
            _download_segmentation_example_file()

        mock_download.assert_not_called()

    def test_returns_path_in_correct_subdirectory(self, mock_data_dir: Path) -> None:
        """Test that returned path is in segmentation_cytosol_example subdirectory."""
        seg_path = mock_data_dir / "segmentation_cytosol_example"
        seg_path.mkdir(parents=True, exist_ok=True)

        with patch("lmd._utils._get_data_dir", return_value=mock_data_dir):
            result = _download_segmentation_example_file()

        assert result.parent.name == "segmentation_cytosol_example"
