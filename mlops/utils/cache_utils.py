from pathlib import Path
from typing import Set, List

def clear_cache(
    directory: str,
    keep_ids: Set[str],
    keep_extensions: Set[str] = None
) -> List[str]:
    """
    Deletes all files in a directory whose filenames do NOT start with any of the `keep_ids`
    and (optionally) whose extensions are not in `keep_extensions`.

    Args:
        directory (str): Path to the directory to clean.
        keep_ids (Set[str]): Set of file ID prefixes (e.g., "12345") to retain.
        keep_extensions (Set[str], optional): Extensions to retain (e.g., {"npz"}).
            If None, all extensions are allowed.

    Returns:
        List[str]: List of deleted filenames.
    """
    directory_path = Path(directory)
    deleted_files = []

    # Normalize and strip keep_ids
    keep_ids = set(str(x).strip() for x in keep_ids)

    for file_path in directory_path.glob("*"):
        if file_path.is_file():
            file_stem = file_path.stem  # e.g., "12345_chunk0"
            file_stem = file_stem.strip()
            file_ext = file_path.suffix.lstrip(".").lower()  # e.g., "npz"

            # Check: does this file start with any of the keep_ids?
            matches_prefix = any(file_stem.startswith(keep_id) for keep_id in keep_ids)
            matches_extension = (keep_extensions is None) or (file_ext in keep_extensions)

            if not matches_prefix or not matches_extension:
                file_path.unlink()
                deleted_files.append(file_path.name)

    remaining_files = list(directory_path.glob('*'))
    print(f"🧹 Deleted {len(deleted_files)} files from {directory}")
    print(f"📋 {len(remaining_files)} files remaining in {str(directory_path)}")
    return deleted_files