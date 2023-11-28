"""Main entry point for CLI application."""
import argparse
import pathlib

import ukis_sat2h5.conversion
import ukis_sat2h5.tiling


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    # top level parser
    parser = argparse.ArgumentParser(
        prog="ukis_sat2h5",
        description="Convert geographic images to H5 file and back and tile them for Deep Learning"
        " applications",
    )
    subparsers = parser.add_subparsers(help="Tools", dest="command")

    # images to H5
    parser_img_to_h5 = subparsers.add_parser(
        "img_to_h5", help="Convert geographic images to H5 arrays"
    )
    parser_img_to_h5.add_argument(
        "-r", "--root_folder", type=pathlib.Path, help="Output H5 file", required=True
    )
    parser_img_to_h5.add_argument(
        "-d", "--dst_file", type=pathlib.Path, help="Output H5 file", required=True
    )
    parser_img_to_h5.add_argument(
        "-f",
        "--file_glob",
        help="File glob to used to recursively find image files in root_folder,"
        ' like "*.tif" or "*.vrt"',
        required=True,
    )
    parser_img_to_h5.add_argument(
        "-l",
        "--label_glob",
        help='File glob to recursively find label files in root_folder, like "*label*"',
        required=True,
    )
    parser_img_to_h5.add_argument(
        "-b",
        "--bands",
        nargs="+",
        type=int,
        default=None,
        help="Indexes of image bands, indexed by 1, like 3 4 5",
    )

    # H5 to images
    parser_h5_to_img = subparsers.add_parser(
        "h5_to_img",
        help="Convert H5 arrays (created with this tool) to back to geographic images",
    )
    parser_h5_to_img.add_argument(
        "-s",
        "--src_file",
        type=pathlib.Path,
        help="Source file to be converted to images",
        required=True,
    )
    parser_h5_to_img.add_argument(
        "-d",
        "--dst_folder",
        type=pathlib.Path,
        help="Target folder for image files",
        required=True,
    )
    parser_h5_to_img.add_argument(
        "-i",
        "--index",
        nargs="*",
        type=int,
        default=None,
        help="Indexes of image bands, indexed by 1.",
    )

    # image tiling options
    parser_tile_h5 = subparsers.add_parser(
        "tile_h5",
        help="Tile H5 array and preserve geographic location",
    )
    parser_tile_h5.add_argument(
        "-s", "--src_file", type=pathlib.Path, help="Source H5 file to be tiled", required=True
    )
    parser_tile_h5.add_argument(
        "-d",
        "--dst_file",
        type=pathlib.Path,
        help="Destination H5 file for tiled arrays",
        required=True,
    )
    parser_tile_h5.add_argument(
        "-t",
        "--tile_size",
        type=int,
        help="Size of tiled arrays in pixels",
        required=True,
    )
    parser_tile_h5.add_argument(
        "-o",
        "--overlap",
        type=int,
        help="Overlap between tiles in pixels",
        required=True,
    )
    parser_tile_h5.add_argument(
        "-a",
        "--target_size",
        type=int,
        help="Target size to which images will be padded",
        required=True,
    )
    parser_tile_h5.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        help="Chunk size of output /img and /lbl datasets in output h5 file. "
        "Setting the value too low can result in high memory consumption. Default: 1",
    )

    return parser.parse_args(args)


def main() -> int:
    """Encapsulate entry point."""
    args = parse_args()

    if args.command == "img_to_h5":
        if not args.root_folder.exists():
            print(  # noqa: T201
                f"Root folder (-r / --root_folder {args.root_folder}) does not exist. Stopping..."
            )
            return 1
        if args.dst_file.exists():
            print(  # noqa: T201
                f"\nDestination file (-d / --dst_file {args.dst_file}) exists. Please (re)move file"
                " or use another destination. Stopping...",
            )
            return 1
        ukis_sat2h5.conversion.convert_img_to_h5(
            args.root_folder, args.dst_file, args.file_glob, args.label_glob, args.bands
        )
    elif args.command == "h5_to_img":
        if not args.src_file.exists():
            print(  # noqa: T201
                f"Source file (-s / --src_file {args.root_folder}) does not exist. Stopping..."
            )
            return 1
        ukis_sat2h5.conversion.convert_h5_to_img(args.src_file, args.dst_folder, args.index)
    elif args.command == "tile_h5":
        if args.dst_file.exists():
            print(  # noqa: T201
                f"\nDestination file (-d / --dst_file {args.dst_file}) exists. Please (re)move file"
                " or use another destination. Stopping...",
            )
            return 1
        ukis_sat2h5.tiling.tile_img_h5(
            args.src_file, args.dst_file, args.tile_size, args.overlap, args.target_size
        )
    else:
        parse_args(["-h"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
