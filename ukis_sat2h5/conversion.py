"""Convert an image folder to h5 array."""
import multiprocessing
import pathlib
import warnings
from glob import glob
from multiprocessing import cpu_count

import dask.array as da
import h5py  # type: ignore
import numpy as np
import rasterio as rio  # type: ignore
from dask.delayed import delayed
from tqdm import tqdm  # type: ignore

_IMG_SIDE_RATIO_WRNG_THRS = 0.5


def convert_img_to_h5(
    src_path: pathlib.Path,
    dst_file: pathlib.Path,
    file_glob: str,
    label_glob: str = "*label*",
    image_bands: list[int] | None = None,
) -> None:
    """Convert an image folder to h5 array.

    Parameters
    ----------
    src_path : pathlib.Path
        Directory in which image files will be recursively searched using `file_glob`.
    dst_file : pathlib.Path
        Destination h5 file into which dataset will be written
    file_glob : str
        File globs to identify image files, e.g. "*.tif" or "*.vrt"
    label_glob : str
        File name pattern used to identify label images. See `pathlib.PurePath.match`
    image_bands : list[int] | None
        Select a subset of image bands by index (starting with 1, see
        https://rasterio.readthedocs.io/en/latest/quickstart.html#reading-raster-data). By default
        (`None`), all bands will be selected.
    """
    # list all images and filter label images if they have the same file_glob
    l_images = [
        i
        for i in _rglob_with_links(src_path.expanduser(), file_glob)
        if not i.match(label_glob)
        # TODO: Python 3.13 replace with src_path.expanduser().rglob(file_glob)  # noqa: FIX002
    ]
    # find labels in the folders where images would be located
    l_labels = [j for i in l_images for j in i.parent.glob(label_glob)]

    assert len(l_images) == len(
        l_labels
    ), f"Number of images ({len(l_images)}) differs from number of labels ({len(l_labels)})"

    # load metadata
    with multiprocessing.Pool(cpu_count()) as pool:
        metadata = list(tqdm(pool.imap(_derive_metadata, l_images), desc="Load img metadata:"))

    widths, heights, epsgs, affines, bands = zip(*metadata, strict=True)

    # all images will be padded. Warn if the images differ greatly in size
    if (min(widths) / max(widths) < _IMG_SIDE_RATIO_WRNG_THRS) or min(heights) / max(
        heights
    ) < _IMG_SIDE_RATIO_WRNG_THRS:
        warnings.warn(  # noqa: B028
            (
                "Some images are significantly smaller than others. "
                "All images will be padded to the maximum mutial extent."
            ),
            UserWarning,
        )

    # check number bands
    if image_bands:
        assert len(set(bands)) == 1, "Number of bands differ across images."
        # check if image_bands fits number of bands
        assert max(bands) >= max(
            image_bands
        ), f"{image_bands=} uses band indices higher than number of bands (= {max(bands)})."

    else:
        image_bands = list(range(1, max(bands) + 1))

    img_arr_del_lst = [
        da.from_delayed(
            delayed(_load_img)(p, max(widths), max(heights), bands=image_bands),
            shape=(len(image_bands), max(heights), max(widths)),
            dtype="uint16",
        )
        for p in l_images
    ]
    lbl_arr_del_lst = [
        da.from_delayed(
            delayed(_load_lbl)(p, max(widths), max(heights)),
            shape=(max(heights), max(widths)),
            dtype="uint16",
        )
        for p in l_labels
    ]

    img_arr = da.stack(img_arr_del_lst)
    lbl_arr = da.stack(lbl_arr_del_lst)

    # shorten paths and convert to string array
    shortened_paths = [p.stem for p in l_images]
    max_str_len = max([len(str(p)) for p in shortened_paths])
    string_dtype = h5py.string_dtype("utf-8", max_str_len)
    path_arr = da.from_array(np.array([str(sp) for sp in shortened_paths], dtype=string_dtype))

    epsg_arr = da.from_array(np.array(epsgs))
    affine_arr = da.from_array(np.array(affines))

    da.to_hdf5(
        dst_file.expanduser(),
        {
            "/img": img_arr,
            "/lbl": lbl_arr,
            "/path": path_arr,
            "/epsg": epsg_arr,
            "/affine": affine_arr,
        },
        compression="lzf",
    )

    _compute_statistics(dst_file.expanduser())


def convert_h5_to_img(
    src_file: pathlib.Path, dst_folder: pathlib.Path, index: int | list[int] | None = None
) -> None:
    """Convert h5 files back to image(s).

    Parameters
    ----------
    src_file : pathlib.Path
        Source file from which the data should be converted into images.
    dst_folder : pathlib.Path
        Destination folder images will be saved to.
    index : int | None
        Index of the array that should be converted to image.
    """
    # create destination folder if missing
    dst_folder.expanduser().mkdir(exist_ok=True)

    if isinstance(index, int):
        _create_raster(index, src_file, dst_folder)
    else:
        if not isinstance(index, list):
            with h5py.File(src_file.expanduser(), mode="r") as f:
                index_list = list(range(f["/path"].shape[0]))
        else:
            index_list = index

        # create rasters using multiprocessing
        with multiprocessing.Pool(cpu_count()) as pool:
            _ = list(
                tqdm(
                    pool.imap_unordered(
                        _create_raster_mp,
                        [(i, src_file, dst_folder) for i in index_list],
                    ),
                    total=len(index_list),
                )
            )


def _create_raster_mp(args: tuple[int, pathlib.Path, pathlib.Path]) -> None:
    _create_raster(*args)


def _create_raster(index: int, src_file: pathlib.Path, dst_folder: pathlib.Path) -> None:
    with h5py.File(src_file.expanduser(), mode="r") as f:
        img = f["/img"][index]
        lbl = f["/lbl"][index]
        path = f["/path"][index]
        epsg = f["/epsg"][index]
        affine = f["/affine"][index]

        metadata = {
            "driver": "GTiff",
            "dtype": "uint16",
            "nodata": 0,
            "count": img.shape[0],
            "height": img.shape[1],
            "width": img.shape[2],
            "crs": rio.CRS.from_epsg(epsg),
            "transform": rio.Affine.from_gdal(*affine),
            "compress": "LZW",
        }
        with rio.open(dst_folder / f"{path.decode('UTF-8')}.tif", "w", **metadata) as dst:
            dst.write(img.astype(rio.uint16))

        metadata.update({"count": 1})
        if len(lbl.shape) == 2:  # noqa: PLR2004
            lbl = lbl[np.newaxis,]
        with rio.open(dst_folder / f"{path.decode('UTF-8')}_label.tif", "w", **metadata) as dst:
            dst.write(lbl.astype(rio.uint16))


def _derive_metadata(image_path: pathlib.Path) -> tuple[int, int, int, tuple, int]:
    with rio.open(image_path, "r") as src:
        width = src.width
        height = src.height
        epsg = src.crs.to_epsg()
        affine = src.transform.to_gdal()
        bands = src.count
    return (width, height, epsg, affine, bands)


def _load_img(
    image_file: pathlib.Path, max_width: int, max_height: int, bands: list[int]
) -> da.Array:
    """Prepare image arrays as generator elements."""
    with rio.open(image_file, "r") as src_img:
        pad_height_img = max_height - src_img.height
        pad_width_img = max_width - src_img.width

        # rasterio returns (channel, height, width), a.k.a. (bands, rows, cols)
        # pytorch tensors are ([batch], channel, height, width); no reshape necessary!
        return da.pad(
            da.from_array(src_img.read(bands)),
            pad_width=((0, 0), (0, pad_height_img), (0, pad_width_img)),
            # TODO: refactor mode to CLI option, to add "reflect"  # noqa: FIX002
            # however this requires the x and y dimension assertions to be more concise as
            # reflections fail with when, e.g. twice the amount of pixels is needed..
            # TODO: in case of "reflect" a valid-pixels-mask should be exportet, too  # noqa: FIX002
            mode="constant",
            # TODO: Refactor constant values to CLI option  # noqa: FIX002
            constant_values=0,
        )


def _load_lbl(image_file: pathlib.Path, max_width: int, max_height: int) -> da.Array:
    """Prepare label arrays as generator elements."""
    with rio.open(image_file.parent / "label.tif", "r") as src_lbl:
        pad_height_lbl = max_height - src_lbl.height
        pad_width_lbl = max_width - src_lbl.width

        return da.pad(
            da.from_array(src_lbl.read(1)),
            pad_width=((0, pad_height_lbl), (0, pad_width_lbl)),
            # TODO: see above in _load_img  # noqa: FIX002
            mode="constant",
            constant_values=0,
        )


def _compute_statistics(h5_file: pathlib.Path) -> None:
    """Compute band statistics across all three split h5-files."""
    with h5py.File(h5_file, "a") as f:
        arr = da.from_array(f["img"])

        stds = da.std(arr, axis=(0, 2, 3)).compute()
        means = da.mean(arr, axis=(0, 2, 3)).compute()

        f.create_dataset("/img_means", data=means)
        f.create_dataset("/img_stds", data=stds)


def _rglob_with_links(path: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    """Recursive globbing fix since pathlib.Path().rglob() does not support symlinks."""
    # TODO: Python 3.13: should be obsolete  # noqa: FIX002
    #       see https://github.com/python/cpython/issues/77609
    return [pathlib.Path(p) for p in glob(f"{path}/**/{pattern}", recursive=True)]  # noqa: PTH207


if __name__ == "__main__":
    # convert_img_to_h5(  # noqa: ERA001,RUF100
    #     src_path=pathlib.Path("~/tmp/sen2_ref_images/"),  # noqa: ERA001
    #     dst_file=pathlib.Path("~/tmp/images.h5"),  # noqa: ERA001
    #     file_glob="*.vrt",  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    # convert_h5_to_img(pathlib.Path("~/tmp/images.h5"), pathlib.Path("~/tmp/images_converted_back/"))  # noqa: ERA001
    # convert_h5_to_img(  # noqa: ERA001,RUF100
    #     pathlib.Path("~/tmp/images.h5"), pathlib.Path("~/tmp/images_converted_back/"), index=0  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    # convert_h5_to_img(  # noqa: ERA001,RUF100
    #     pathlib.Path("~/tmp/images.h5"), pathlib.Path("~/tmp/images_converted_back/"), index=213  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    # convert_h5_to_img(  # noqa: ERA001,RUF100
    #     pathlib.Path("~/tmp/images_tiled.h5"),  # noqa: ERA001
    #     pathlib.Path("~/tmp/images_converted_back/"),  # noqa: ERA001
    #     index=100000,  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    # convert_h5_to_img(
    #     pathlib.Path("~/tmp/images_tiled.h5"), pathlib.Path("~/tmp/images_converted_back/")  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    pass
