from typing import BinaryIO, Literal

from aiogram import Bot

from io import BytesIO

import av
from PIL import Image
from PIL.Image import Image as ImageObject
from lottie.utils.stripper import float_strip
from lottie.importers import importers
from lottie.exporters import exporters

from aiogram.types import File
from google.generativeai.protos import Blob


class Media:
    """
    A class for handling media processing, including downloading files from Telegram,
    converting animated stickers (TGS and WEBM) to PNG format, and more.
    """

    @staticmethod
    def _blob(mime: str, data: bytes) -> Blob:
        """
        Create a Blob object with the specified MIME type and data.

        Args:
            mime (str): The MIME type of the data (e.g., 'image/png', 'application/json').
            data (bytes): The raw binary data to be included in the Blob.

        Returns:
            Blob: A Blob object containing the specified MIME type and data.
        """
        # noinspection PyTypeChecker
        return Blob(mime_type=mime, data=data)

    @staticmethod
    def _tgs(in_file: BytesIO | BinaryIO) -> bytes:
        """
        Process a TGS (Lottie) animation file and convert it to PNG format.

        Args:
            in_file (BytesIO | BinaryIO): Input TGS file in binary format.

        Returns:
            bytes: The raw PNG image data as bytes.
        """
        importer = next((p for p in importers if "tgs" in p.extensions), None)
        exporter = exporters.get_from_filename("_.png")
        animation = importer.process(in_file)
        float_strip(animation)

        output_buffer = BytesIO()
        exporter.process(animation, output_buffer, frame=1)

        return output_buffer.getvalue()

    @staticmethod
    def _webm(in_file: BytesIO | BinaryIO) -> bytes:
        """
        Extract the first frame of a WEBM video and convert it to PNG format.

        Args:
            in_file (BytesIO | BinaryIO): Input WEBM video in binary format.

        Returns:
            bytes: The raw PNG image data as bytes.
        """
        container = av.open(in_file)
        png_output = BytesIO()

        for frame in container.decode(video=0):
            rgb_frame = frame.to_image()
            rgb_frame.save(png_output, format='PNG')
            png_output.seek(0)
            break

        return png_output.getvalue()

    @classmethod
    def _handle_sticker(cls, mime: Literal["sticker/tgs", "sticker/webm"], file_data: BytesIO | BinaryIO) -> Blob | None:
        """
        Handle sticker processing by converting animated TGS and WEBM formats to PNG.

        Args:
            mime (Literal["sticker/tgs", "sticker/webm"]): MIME type indicating sticker format.
            file_data (BytesIO | BinaryIO): Raw sticker file data.

        Returns:
            Blob | None: A Blob object containing the processed PNG image data, or None if unrecognized.
        """
        mime_methods = {
            "sticker/tgs": cls._tgs,
            "sticker/webm": cls._webm
        }

        method = mime_methods.get(mime)
        if method:
            return cls._blob("image/png", method(file_data))

        return None

    @classmethod
    async def download(cls, bot: Bot, file_id: str, mime: str | None = None) -> ImageObject | Blob | None:
        """
        Download and process media from a given Telegram file ID. Handles both images and stickers (TGS/WEBM).

        Args:
            bot (Bot): The Telegram bot instance.
            file_id (str): The Telegram file ID to download.
            mime (str, optional): The custom MIME type for processing. Defaults to None.

        Returns:
            ImageObject | Blob | None:
                - If the file is an image, returns a PIL Image object.
                - If the file is a sticker, returns a Blob object with PNG data.
                - Returns None if the file type is unsupported.
        """
        file: File = await bot.get_file(file_id)
        file_data = await bot.download_file(file.file_path)

        if not file_data:
            return None

        if mime in {"sticker/tgs", "sticker/webm"}:
            mime: Literal["sticker/tgs", "sticker/webm"]
            return cls._handle_sticker(mime, file_data)

        if mime:
            return cls._blob(mime, file_data.getvalue())

        return Image.open(file_data)
