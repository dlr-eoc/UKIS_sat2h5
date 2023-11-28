"""Test contents of ukis_sat2h5/tiling.py."""
import pathlib
import shutil

import rasterio as rio  # type: ignore

from ukis_sat2h5 import conversion, tiling


def test_single_image_tiled(single_image_folder: pathlib.Path) -> None:
    """Test the conversion and tiling of a single image by comparing the input and tiled output."""
    tile_size = 256
    h5_file = single_image_folder / "img.h5"
    conversion.convert_img_to_h5(
        src_path=single_image_folder,
        dst_file=h5_file,
        file_glob="*.tif",
        label_glob="*label*",
        image_bands=[1, 2, 3, 4],
    )
    h5_file_tiled = single_image_folder / "img_tiled.h5"
    tiling.tile_img_h5(h5_file, h5_file_tiled, tile_size=tile_size, overlap=128, target_size=1024)

    roundtrip_image_folder = single_image_folder / "img_tiled_roundtrip"
    conversion.convert_h5_to_img(
        src_file=h5_file_tiled, dst_folder=roundtrip_image_folder, index=None
    )

    with rio.open(single_image_folder / "img.tif", mode="r") as original_image:
        for i in roundtrip_image_folder.rglob("img*.tif"):
            if i.match("*label.tif"):
                continue
            with rio.open(i, mode="r") as test_image:
                assert test_image.width == tile_size
                assert test_image.height == tile_size
                # read data
                test_image_arr = test_image.read()
                original_image_arr = original_image.read(
                    window=rio.windows.from_bounds(
                        *test_image.bounds, transform=original_image.transform
                    )
                )

                # As images can be padded during conversion, shapes might differ
                if original_image_arr.shape != test_image_arr.shape:
                    b, x, y = original_image_arr.shape
                    # check number of bands matches (should be untouched by padding)
                    assert b == test_image_arr.shape[0]
                    # check if everything outside was set to 0, the chosen `constant_values`
                    assert (test_image_arr[:, x:, y:] == 0).all()

                    # reshape image
                    test_image_arr = test_image_arr[:, :x, :y]

                assert (test_image_arr == original_image_arr).all()

    # same test but for label
    with rio.open(single_image_folder / "label.tif", mode="r") as original_image:
        for i in roundtrip_image_folder.rglob("img*.tif"):
            if not i.match("*label.tif"):
                continue
            with rio.open(i, mode="r") as test_image:
                assert test_image.width == tile_size
                assert test_image.height == tile_size
                # read data
                test_image_arr = test_image.read()
                original_image_arr = original_image.read(
                    window=rio.windows.from_bounds(
                        *test_image.bounds, transform=original_image.transform
                    )
                )

                # As images can be padded during conversion, shapes might differ
                if original_image_arr.shape != test_image_arr.shape:
                    b, x, y = original_image_arr.shape
                    # check number of bands matches (should be untouched by padding)
                    assert b == test_image_arr.shape[0]
                    # check if everything outside was set to 0, the chosen `constant_values`
                    assert (test_image_arr[:, x:, y:] == 0).all()
                    test_image_arr = test_image_arr[:, :x, :y]

                assert (test_image_arr == original_image_arr).all()

    # cleanup
    h5_file.unlink()
    h5_file_tiled.unlink()
    shutil.rmtree(roundtrip_image_folder)
