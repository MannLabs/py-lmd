from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Literal

import requests
from filelock import FileLock
from tqdm import tqdm


def _get_data_dir() -> Path:
    """Get path to data that was packaged with scPortrait.

    Returns:
        Path to data directory
    """

    def find_root_by_folder(marker_folder: str, current_path: Path) -> Path | None:
        for parent in current_path.parents:
            if (parent / marker_folder).is_dir():
                return parent
        return None

    src_code_dir = find_root_by_folder("lmd", Path(__file__))

    if src_code_dir is None:
        raise FileNotFoundError("Could not find pyLMD source directory")

    data_dir = src_code_dir / "pylmd_data"

    return data_dir.absolute()


def _download(
    url: str,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
) -> None:  # pragma: no cover
    """Downloads a file irrespective of format.

    Args:
        url: URL to download.
        archive_format: The format if an archive file.
        output_file_name: Name of the downloaded file.
        output_path: Path to download/extract the files to. Defaults to 'OS tmpdir' if not specified.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"pylmd_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    def _sanitize_file_name(file_name):
        if os.name == "nt":
            file_name = file_name.replace("?", "_").replace("*", "_")
        return file_name

    download_to_path = Path(
        _sanitize_file_name(
            f"{output_path}{output_file_name}"
            if str(output_path).endswith("/")
            else f"{output_path}/{output_file_name}"
        )
    )

    Path(output_path).mkdir(parents=True, exist_ok=True)
    lock_path = f"{download_to_path}.lock"

    with FileLock(lock_path):
        if download_to_path.exists():
            warning = f"File {download_to_path} already exists!"
            if not overwrite:
                print(warning)
                Path(lock_path).unlink()
                return
            else:
                print(f"{warning} Overwriting...")

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        temp_file_name = f"{download_to_path}.part"

        with (
            open(temp_file_name, "wb") as file,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading...") as progress_bar,
        ):
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

        Path(temp_file_name).replace(download_to_path)

        if archive_format:
            shutil.unpack_archive(download_to_path, output_path, format=archive_format)
            os.remove(download_to_path)
        else:
            download_to_path.with_name(output_file_name)

    Path(lock_path).unlink()


def _download_glyphs() -> Path:
    """Download and extract the Glyphs.

    Returns:
        Path to the downloaded and extracted Glyphs
    """
    data_dir = Path(_get_data_dir())
    save_path = data_dir / "glyphs"

    if not save_path.exists():
        _download(
            url="https://zenodo.org/records/14623414/files/glyphs.zip?download=1",
            output_path=str(save_path),
            archive_format="zip",
        )
    else:
        # check that directory is not empty
        if not any(save_path.iterdir()):
            _download(
                url="https://zenodo.org/records/14623414/files/glyphs.zip?download=1",
                output_path=str(save_path),
                archive_format="zip",
            )

    return save_path


def _download_segmentation_example_file() -> Path:
    """Download example segmentation file.

    Returns:
        Path to the downloaded and extracted tiff file.
    """

    data_dir = Path(_get_data_dir())
    save_path = data_dir / "segmentation_cytosol_example"

    output_file_name = "segmentation_mask.tif"
    if not save_path.exists():
        _download(
            url="https://zenodo.org/records/15681419/files/segmentation_cytosol.tiff?download=1",
            output_path=str(save_path),
            output_file_name=output_file_name,
            archive_format=None,
        )

    return save_path / output_file_name
