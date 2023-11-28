"""Test contents of ukis_sat2h5/conversion.py."""
import pathlib
import shutil

import numpy as np
import pytest
import rasterio as rio  # type: ignore

from tests.conftest import _create_image
from ukis_sat2h5 import conversion

SINGLE_IMAGE_FOLDER_IMAGE_WIDTH = 999
SINGLE_IMAGE_FOLDER_IMAGE_HEIGHT = 999
SINGLE_IMAGE_FOLDER_IMAGE_EPSG = 32601
SINGLE_IMAGE_FOLDER_IMAGE_BANDS = 4


def test_single_img_to_h5_to_img(single_image_folder: pathlib.Path) -> None:
    """Test conversion to and from h5 results in identical raster."""
    # convert one image to h5
    h5_file = single_image_folder / "img.h5"
    conversion.convert_img_to_h5(
        src_path=single_image_folder,
        dst_file=h5_file,
        file_glob="*.tif",
        label_glob="*label*",
    )

    # convert h5 back to image
    roundtrip_image_folder = single_image_folder / "img_roundtrip"
    conversion.convert_h5_to_img(src_file=h5_file, dst_folder=roundtrip_image_folder, index=None)

    # open both arrays and check whether they match
    with (
        rio.open(single_image_folder / "img.tif", "r") as original_image,
        rio.open(single_image_folder / "label.tif", "r") as original_label,
        rio.open(roundtrip_image_folder / "img.tif", "r") as roundtrip_image,
        rio.open(roundtrip_image_folder / "img_label.tif", "r") as roundtrip_label,
    ):
        arr_img_original = original_image.read()
        arr_lbl_original = original_label.read()
        arr_img_roundtrip = roundtrip_image.read()
        arr_lbl_roundtrip = roundtrip_label.read()

        assert (arr_img_original == arr_img_roundtrip).all()
        assert (arr_lbl_original == arr_lbl_roundtrip).all()

        assert original_image.meta["width"] == roundtrip_image.meta["width"]
        assert original_image.meta["height"] == roundtrip_image.meta["height"]
        assert original_image.meta["transform"] == roundtrip_image.meta["transform"]
        assert original_image.meta["crs"] == roundtrip_image.meta["crs"]

        assert original_label.meta["width"] == roundtrip_label.meta["width"]
        assert original_label.meta["height"] == roundtrip_label.meta["height"]
        assert original_label.meta["transform"] == roundtrip_label.meta["transform"]
        assert original_label.meta["crs"] == roundtrip_label.meta["crs"]

    # cleanup
    h5_file.unlink()
    shutil.rmtree(roundtrip_image_folder)


def test_multiple_image_to_h5_to_img(multiple_image_folder: pathlib.Path) -> None:
    """Test conversion to and from h5 results in identical raster."""
    # convert one image to h5
    h5_file = multiple_image_folder / "img.h5"
    conversion.convert_img_to_h5(
        src_path=multiple_image_folder,
        dst_file=h5_file,
        file_glob="*.tif",
        label_glob="*label*",
        image_bands=[1, 2, 3, 4],
    )

    # convert h5 back to image
    roundtrip_image_folder = multiple_image_folder / "img_roundtrip"
    conversion.convert_h5_to_img(src_file=h5_file, dst_folder=roundtrip_image_folder, index=None)

    # open both arrays and check whether they match
    for i in range(9):
        with (
            rio.open(multiple_image_folder / f"image_{i}" / f"img_{i}.tif", "r") as original_image,
            rio.open(multiple_image_folder / f"image_{i}" / "label.tif", "r") as original_label,
            rio.open(roundtrip_image_folder / f"img_{i}.tif", "r") as roundtrip_image,
            rio.open(roundtrip_image_folder / f"img_{i}_label.tif", "r") as roundtrip_label,
        ):
            arr_img_original = original_image.read()
            arr_lbl_original = original_label.read()
            arr_img_roundtrip = roundtrip_image.read()
            arr_lbl_roundtrip = roundtrip_label.read()

            assert (arr_img_original == arr_img_roundtrip).all()
            assert (arr_lbl_original == arr_lbl_roundtrip).all()

            assert original_image.meta["width"] == roundtrip_image.meta["width"]
            assert original_image.meta["height"] == roundtrip_image.meta["height"]
            assert original_image.meta["transform"] == roundtrip_image.meta["transform"]
            assert original_image.meta["crs"] == roundtrip_image.meta["crs"]

            assert original_label.meta["width"] == roundtrip_label.meta["width"]
            assert original_label.meta["height"] == roundtrip_label.meta["height"]
            assert original_label.meta["transform"] == roundtrip_label.meta["transform"]
            assert original_label.meta["crs"] == roundtrip_label.meta["crs"]

    # cleanup
    h5_file.unlink()
    shutil.rmtree(roundtrip_image_folder)


def test_single_index_conversion(multiple_image_folder: pathlib.Path) -> None:
    """Test conversion to and from h5 results in identical raster with a single index."""
    # convert one image to h5
    h5_file = multiple_image_folder / "img.h5"
    conversion.convert_img_to_h5(
        src_path=multiple_image_folder,
        dst_file=h5_file,
        file_glob="*.tif",
        label_glob="*label*",
        image_bands=[1, 2, 3, 4],
    )

    # convert h5 back to image
    roundtrip_image_folder = multiple_image_folder / "img_roundtrip"
    conversion.convert_h5_to_img(src_file=h5_file, dst_folder=roundtrip_image_folder, index=5)

    with (
        rio.open(multiple_image_folder / "image_5" / "img_5.tif", "r") as original_image,
        rio.open(multiple_image_folder / "image_5" / "label.tif", "r") as original_label,
        rio.open(roundtrip_image_folder / "img_5.tif", "r") as roundtrip_image,
        rio.open(roundtrip_image_folder / "img_5_label.tif", "r") as roundtrip_label,
    ):
        arr_img_original = original_image.read()
        arr_lbl_original = original_label.read()
        arr_img_roundtrip = roundtrip_image.read()
        arr_lbl_roundtrip = roundtrip_label.read()

        assert (arr_img_original == arr_img_roundtrip).all()
        assert (arr_lbl_original == arr_lbl_roundtrip).all()

        assert original_image.meta["width"] == roundtrip_image.meta["width"]
        assert original_image.meta["height"] == roundtrip_image.meta["height"]
        assert original_image.meta["transform"] == roundtrip_image.meta["transform"]
        assert original_image.meta["crs"] == roundtrip_image.meta["crs"]

        assert original_label.meta["width"] == roundtrip_label.meta["width"]
        assert original_label.meta["height"] == roundtrip_label.meta["height"]
        assert original_label.meta["transform"] == roundtrip_label.meta["transform"]
        assert original_label.meta["crs"] == roundtrip_label.meta["crs"]

    # cleanup
    h5_file.unlink()
    shutil.rmtree(roundtrip_image_folder)


def test_index_list_conversion(multiple_image_folder: pathlib.Path) -> None:
    """Test conversion to and from h5 results in identical raster with a single index."""
    # convert one image to h5
    h5_file = multiple_image_folder / "img.h5"
    conversion.convert_img_to_h5(
        src_path=multiple_image_folder,
        dst_file=h5_file,
        file_glob="*.tif",
        label_glob="*label*",
        image_bands=[1, 2, 3, 4],
    )

    # convert h5 back to image
    idx_lst = [4, 5]
    roundtrip_image_folder = multiple_image_folder / "img_roundtrip"
    conversion.convert_h5_to_img(src_file=h5_file, dst_folder=roundtrip_image_folder, index=idx_lst)

    for i in idx_lst:
        with (
            rio.open(multiple_image_folder / f"image_{i}" / f"img_{i}.tif", "r") as original_image,
            rio.open(multiple_image_folder / f"image_{i}" / "label.tif", "r") as original_label,
            rio.open(roundtrip_image_folder / f"img_{i}.tif", "r") as roundtrip_image,
            rio.open(roundtrip_image_folder / f"img_{i}_label.tif", "r") as roundtrip_label,
        ):
            arr_img_original = original_image.read()
            arr_lbl_original = original_label.read()
            arr_img_roundtrip = roundtrip_image.read()
            arr_lbl_roundtrip = roundtrip_label.read()

            assert (arr_img_original == arr_img_roundtrip).all()
            assert (arr_lbl_original == arr_lbl_roundtrip).all()

            assert original_image.meta["width"] == roundtrip_image.meta["width"]
            assert original_image.meta["height"] == roundtrip_image.meta["height"]
            assert original_image.meta["transform"] == roundtrip_image.meta["transform"]
            assert original_image.meta["crs"] == roundtrip_image.meta["crs"]

            assert original_label.meta["width"] == roundtrip_label.meta["width"]
            assert original_label.meta["height"] == roundtrip_label.meta["height"]
            assert original_label.meta["transform"] == roundtrip_label.meta["transform"]
            assert original_label.meta["crs"] == roundtrip_label.meta["crs"]

    # cleanup
    h5_file.unlink()
    shutil.rmtree(roundtrip_image_folder)


def test__derive_metadata(single_image_folder: pathlib.Path) -> None:
    """Test metadata derivation separately, as it is covered by multiprocessing."""
    width, height, epsg, affine, bands = conversion._derive_metadata(
        single_image_folder / "img.tif"
    )

    assert width == SINGLE_IMAGE_FOLDER_IMAGE_WIDTH
    assert height == SINGLE_IMAGE_FOLDER_IMAGE_HEIGHT
    assert epsg == SINGLE_IMAGE_FOLDER_IMAGE_EPSG
    assert affine == (0.0, 10.0, 0.0, 0.0, 0.0, -10.0)
    assert bands == SINGLE_IMAGE_FOLDER_IMAGE_BANDS


def test__load_img(single_image_folder: pathlib.Path) -> None:
    """Test loading images into dask arrays, as it might be covered by dask multiprocessing."""
    da_img_arr = conversion._load_img(
        single_image_folder / "img.tif", max_width=1024, max_height=1024, bands=[2, 3]
    )

    assert da_img_arr.shape == (2, 1024, 1024)


def test__load_lbl(single_image_folder: pathlib.Path) -> None:
    """Test loading images into dask arrays, as it might be covered by dask multiprocessing."""
    da_lbl_arr = conversion._load_lbl(
        single_image_folder / "img.tif", max_width=1024, max_height=1024
    )

    assert da_lbl_arr.shape == (1024, 1024)


def test_warning_when_images_are_of_varying_size(multiple_image_folder: pathlib.Path) -> None:
    """A warning should be emitted when images are of vastly differing sizes."""
    folder = multiple_image_folder / "image_small_dimensions"
    folder.mkdir(exist_ok=True)
    _create_image(
        folder / "img_small_dimensions.tif",
        min_val=0,
        max_val=255,
        nbands=4,
        nrows=200,
        ncols=200,
        dtype=np.dtype("uint8"),
    )
    _create_image(
        folder / "label.tif",
        min_val=0,
        max_val=255,
        nbands=None,
        nrows=200,
        ncols=200,
        dtype=np.dtype("uint8"),
    )

    with pytest.warns(UserWarning, match="Some images are significantly smaller than others."):
        conversion.convert_img_to_h5(
            multiple_image_folder,
            multiple_image_folder / "arr.h5",
            "*.tif",
            "*label*",
            image_bands=[1, 2, 3, 4],
        )

    shutil.rmtree(folder)
    (multiple_image_folder / "arr.h5").unlink()


def test_assertionerror_when_images_have_varying_number_of_bands(
    multiple_image_folder: pathlib.Path,
) -> None:
    """A warning should be emitted when images are of vastly differing sizes."""
    folder = multiple_image_folder / "image_wrong_number_of_bands"
    folder.mkdir(exist_ok=True)
    _create_image(
        folder / "img_small_dimensions.tif",
        min_val=0,
        max_val=255,
        nbands=2,
        nrows=999,
        ncols=999,
        dtype=np.dtype("uint8"),
    )
    _create_image(
        folder / "label_small_dimensions.tif",
        min_val=0,
        max_val=255,
        nbands=None,
        nrows=999,
        ncols=999,
        dtype=np.dtype("uint8"),
    )

    with pytest.raises(AssertionError):
        conversion.convert_img_to_h5(
            multiple_image_folder,
            multiple_image_folder / "arr.h5",
            "*.tif",
            "*label*",
            image_bands=[1, 2, 3, 4],
        )
    shutil.rmtree(folder)
