"""
Path resolution utilities
"""

import os
from typing import Union


def resolve_data_path(base_dir: str, file_path: Union[str, None]) -> str:
    """
    Resolve file path, handling relative and absolute paths.
    
    Args:
        base_dir: Base directory for relative paths
        file_path: File path (can be absolute, relative, or start with ../)
        
    Returns:
        Resolved absolute path
        
    Raises:
        ValueError: If file_path is None
    """
    if file_path is None:
        raise ValueError("file_path cannot be None")
    
    if os.path.isabs(file_path):
        return file_path
    
    if file_path.startswith('../'):
        return os.path.normpath(os.path.abspath(file_path))
    
    return os.path.normpath(os.path.join(base_dir, file_path))
