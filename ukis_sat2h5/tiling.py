"""Tile an images stored in a h5 file produced by sat2h5.conversion.convert_sat_to_h5."""
import pathlib
from itertools import chain

import dask.array as da
import h5py  # type: ignore
import numpy as np
import rasterio as rio  # type: ignore


def tile_img_h5(
    src_file: pathlib.Path,
    dst_file: pathlib.Path,
    tile_size: int,
    overlap: int,
    target_size: int,
    chunk_size: int | None = 1,
) -> None:
    """Tile image array and provide tiled transforms for h5 files created by ukis_sat2h5.conversion.

    Parameters
    ----------
    src_file : pathlib.Path
        input h5 file containing arrays to be tiled. Ideally created by `ukis_sat2h5.conversion`
    dst_file : pathlib.Path
        output h5 file
    tile_size : int
        size of tiled arrays in pixels
    overlap : int
        overlap between tiles in pixels in x and y directions
    target_size : int
        to which size should the images be expanded? This can be helpful to target an integer
        divider between the array's target size and `tile_size`, e.g. `tile_size=256`, `overlap=128`
        then `target_size` can be set to 1024
    chunk_size : int | None
        Size of chunks for /img and /lbl datasets in output h5_file. Setting chunk_size to low
        values can lead to very high memory consumption.
    """
    with h5py.File(src_file) as f:
        # get items from original file
        img = da.from_array(f["img"], chunks=(1, None, None, None)).astype("int16")
        lbl = da.from_array(f["lbl"], chunks=(1, None, None)).astype("int16")
        lbl = np.expand_dims(lbl, axis=1)
        img_means = da.from_array(f["img_means"]).astype("int16")
        img_stds = da.from_array(f["img_stds"]).astype("int16")
        path = da.from_array(f["path"]).compute()
        epsg = da.from_array(f["epsg"])
        affine = da.from_array(f["affine"]).compute()

        n_tiles = (((target_size - tile_size) // overlap) + 1) ** 2

        img_tiled = da.map_blocks(
            _tile_array,
            img,
            tile_size,
            overlap,
            target_size,
            chunks=(n_tiles, img.shape[1], tile_size, tile_size),
            dtype=img.dtype,
        )
        lbl_tiled = da.map_blocks(
            _tile_array,
            lbl,
            tile_size,
            overlap,
            target_size,
            chunks=(n_tiles, lbl.shape[1], tile_size, tile_size),
            dtype=lbl.dtype,
        )

        # include the corresponding file paths
        path_tiled = _tile_paths(path, n_tiles, img_tiled)
        epsg_tiled = _tile_epsgs(epsg, n_tiles, img_tiled)
        affine_tiled = _tile_affines(affine, target_size, tile_size, overlap, img_tiled)

        # In order to avoid rechunking the array with dask, a more convoluted saving procedure is
        # chosen over the more straight-forward da.to_hdf5(). Using da.store instead, allows
        # chunking to be outsourced to h5py.File.create_dataset.
        with h5py.File(dst_file, mode="a") as dst:
            # image
            img_dset = dst.create_dataset(
                "/img",
                shape=img_tiled.shape,
                chunks=(chunk_size, *img_tiled.shape[1:]),
                dtype=img_tiled.dtype,
                compression="lzf",
                shuffle=True,
            )
            da.store(img_tiled, img_dset)

            # label
            lbl_dset = dst.create_dataset(
                "/lbl",
                shape=lbl_tiled.shape,
                chunks=(chunk_size, *lbl_tiled.shape[1:]),
                dtype=lbl_tiled.dtype,
                compression="lzf",
                shuffle=True,
            )
            da.store(lbl_tiled, lbl_dset)

        # add additional data
        da.to_hdf5(
            dst_file,
            {
                "/img_means": img_means,
                "/img_stds": img_stds,
                "/path": path_tiled,
                "/epsg": epsg_tiled,
                "/affine": affine_tiled,
            },
            compression="lzf",
            chunks=True,
            shuffle=True,
        )


def _tile_array(arr: np.ndarray, tile_size: int, overlap: int, pad_size: int = 1024) -> np.ndarray:
    """Tile an array with overlap.

    First, all input images get padded to target_size for even tiling.

    Parameters
    ----------
    arr : numpy.ndarray
        input array
    tile_size : int
        desired size of tiled array
    overlap : int
        pixels overlapping (if no overlap put tile_size)
    pad_size : int
        size to which the images get padded

    Returns
    -------
    numpy.ndarray
        array of size (tiles, channels, tile_size, tile_size)
    """
    # pad image/label to target_size
    h = arr.shape[-2]
    w = arr.shape[-1]
    c = arr.shape[-3]
    pad_h = pad_size - h
    pad_w = pad_size - w
    padded = np.pad(
        arr, pad_width=((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0
    )

    # use strides for creating windows
    strided = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=(c, tile_size, tile_size), axis=(1, 2, 3)
    )  # type: ignore

    # slice by with distance of overlap
    return strided[:, :, ::overlap, ::overlap].reshape(-1, c, tile_size, tile_size)


def _tile_affines(
    affine: da.Array, target_size: int, tile_size: int, overlap: int, img_tiled: da.Array
) -> da.Array:
    """Derive new affine projection tuples from for each image."""
    affine_tiled = da.from_array(
        np.array(
            list(
                chain.from_iterable(
                    [_tile_one_affine(a, target_size, tile_size, overlap) for a in affine]
                )
            )
        )
    )
    assert (
        img_tiled.shape[0] == affine_tiled.shape[0]
    ), f"{img_tiled.shape[0]=} != {affine_tiled.shape[0]=}"
    return affine_tiled


def _tile_epsgs(epsg: da.Array, n_tiles: int, img_tiled: da.Array) -> da.Array:
    """Replicate the EPSG codes by the number of tiles."""
    epsg_tiled = da.from_array(np.repeat(np.array(epsg), n_tiles))
    assert (
        img_tiled.shape[0] == epsg_tiled.shape[0]
    ), f"{img_tiled.shape[0]=} != {epsg_tiled.shape[0]=}"
    return epsg_tiled


def _tile_paths(path: da.Array, n_tiles: int, img_tiled: da.Array) -> da.Array:
    """Replicate the paths by number of tiles and append a tile index."""
    max_path_chrs = max([len(p.decode("utf-8")) for p in path]) + len(str(n_tiles)) + 1  # + 1 for _
    string_dtype = h5py.string_dtype("utf-8", max_path_chrs)
    path_tiled = da.from_array(
        np.array(
            [f"{p.decode('UTF-8')}_{i:0{len(str(n_tiles))}}" for p in path for i in range(n_tiles)],
            dtype=string_dtype,
        )
    )
    assert (
        img_tiled.shape[0] == path_tiled.shape[0]
    ), f"{img_tiled.shape[0]=} != {path_tiled.shape[0]=}"

    return path_tiled


def _tile_one_affine(
    affine: tuple[float], target_size: int, tile_size: int, overlap: int
) -> list[tuple[float]]:
    """Create affine transforms for all tiles of an image based on target_size and overlap."""
    return [
        _affine_from_rowcol(affine, row, col)
        for row in range(0, target_size - tile_size + 1, overlap)
        for col in range(0, target_size - tile_size + 1, overlap)
    ]


def _affine_from_rowcol(affine: tuple[float], row: int, col: int) -> rio.Affine:
    """Derive a new affine transform from exisitng affine and rows/cols."""
    at = rio.transform.AffineTransformer(rio.Affine.from_gdal(*affine))
    x, y = at.xy(row, col, offset="ul")
    return (x, *affine[1:3], y, *affine[-2:])


if __name__ == "__main__":
    # tile_img_h5(  # noqa: ERA001,RUF100
    #     src_file=pathlib.Path("~/tmp/images.h5").expanduser(),  # noqa: ERA001
    #     dst_file=pathlib.Path("~/tmp/images_tiled.h5").expanduser(),  # noqa: ERA001
    #     tile_size=32,  # noqa: ERA001
    #     overlap=16,  # noqa: ERA001
    #     target_size=1024,  # noqa: ERA001
    # )  # noqa: ERA001,RUF100
    pass
