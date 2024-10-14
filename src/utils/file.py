import os
import inspect
from typing import Callable, Any


class File:

    @staticmethod
    def _get_caller_directory() -> str:
        """
        Get the directory path from where the method was called.

        Returns:
        - str: The directory path of the calling file.
        """
        # Get the caller's frame to determine from where the function was called
        frame = inspect.stack()[2]  # Get the stack frame of the caller of the calling method
        module = inspect.getmodule(frame[0])
        caller_file_path = module.__file__ if module else os.getcwd()

        # Get the directory from where the function was called
        return os.path.dirname(os.path.abspath(caller_file_path))

    @classmethod
    def read(cls, file_name: str, post_processor: Callable[[str], Any] = None) -> Any:
        """
        Read a file's content and optionally apply a post-processor function.

        Args:
        - file_name (str): Name of the file to read.
        - post_processor (Callable[[str], Any], optional): A function to process the content
          after reading (e.g., JSON loader). Defaults to None.

        Returns:
        - Any: The file content, processed or unprocessed.
        """
        caller_dir = cls._get_caller_directory()
        file_path = os.path.join(caller_dir, file_name)

        with open(file_path, 'r') as file:
            content = file.read()
            if post_processor:
                return post_processor(content)
            return content
