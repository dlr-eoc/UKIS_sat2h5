"""Configuration useful across different tests."""
import pathlib
import shutil

import numpy as np
import pytest
import rasterio as rio  # type: ignore


@pytest.fixture(scope="session")
def single_image_folder(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Fixture to reuse image folder with one file across tests."""
    # create file name
    single_image_folder = tmp_path_factory.mktemp("single_image_folder")

    # create image
    img_file_name = single_image_folder / "img.tif"
    _create_image(img_file_name, 0, 255, 4, 999, 999, np.dtype("uint8"))

    # create label
    lbl_file_name = single_image_folder / "label.tif"
    _create_image(lbl_file_name, 0, 3, None, 999, 999, np.dtype("uint8"))

    # return folder name
    yield single_image_folder

    # clean up after session
    shutil.rmtree(single_image_folder)


@pytest.fixture(scope="session")
def multiple_image_folder(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Fixture to reuse image folder with multiple image files across tests."""
    multiple_image_folder = tmp_path_factory.mktemp("multiple_image_folder")
    for i in range(10):
        # create image
        folder = multiple_image_folder / f"image_{i}"
        folder.mkdir()
        img_file_name = folder / f"img_{i}.tif"
        _create_image(img_file_name, 0, 255, 4, 999, 999, np.dtype("uint8"))

        # create label
        lbl_file_name = folder / "label.tif"
        _create_image(lbl_file_name, 0, 3, None, 999, 999, np.dtype("uint8"))

    yield multiple_image_folder

    # clean up after session
    shutil.rmtree(multiple_image_folder)


def _create_image(
    img_file_name: pathlib.Path | str,
    min_val: int,
    max_val: int,
    nbands: int | None,
    nrows: int,
    ncols: int,
    dtype: np.dtype,
) -> None:
    if not nbands:
        nbands = 1
    dims = tuple(i for i in (nbands, nrows, ncols) if i)  # filter None dimensions
    img_arr = np.random.default_rng().integers(min_val, max_val, dims, dtype=dtype)

    meta = {
        "driver": "GTiff",
        "count": nbands,
        "width": ncols,
        "height": nrows,
        "dtype": dtype,
        "compress": "lzw",
        "crs": rio.crs.CRS().from_epsg("32601"),
        "transform": rio.Affine.from_gdal(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    }

    with rio.open(img_file_name, "w", **meta) as dst:
        dst.write(img_arr)
