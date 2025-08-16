import logging
from typing import BinaryIO, Literal

from aiogram import Bot

from io import BytesIO

import av
from PIL import Image
from PIL.Image import Image as ImageObject
from lottie.utils.stripper import float_strip
from lottie.importers import importers
from lottie.exporters import exporters

from aiogram.types import File, Message
from google.generativeai.protos import Blob


class Media:
    """
    A class for handling media processing, including downloading files from Telegram,
    converting animated stickers (TGS and WEBM) to PNG format, video notes, and more.
    """
    MAX_VIDEO_SIZE = 20 * 1024 * 1024 - 1

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

        Raises:
            ValueError: If no suitable importer or exporter is found, or if the conversion fails.
        """
        # Find a suitable importer for TGS files
        importer = next((p for p in importers if "tgs" in p.extensions), None)
        if not importer:
            raise ValueError("No suitable importer found for TGS format")

        # Get PNG exporter
        exporter = exporters.get_from_filename("_.png")
        if not exporter:
            raise ValueError("No suitable exporter found for PNG format")

        try:
            # Process the animation
            animation = importer.process(in_file)
            float_strip(animation)

            # Export to PNG
            output_buffer = BytesIO()
            exporter.process(animation, output_buffer, frame=1)
            return output_buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to convert TGS to PNG: {str(e)}")

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

    @staticmethod
    def _video_note(in_file: BytesIO | BinaryIO) -> bytes:
        """
        Process a Telegram video note (MP4 format).

        Args:
            in_file (BytesIO | BinaryIO): Input video note file in binary format.

        Returns:
            bytes: The raw video data as bytes.
        """
        # For video notes, we'll just return the raw MP4 data
        return in_file.getvalue()

    @classmethod
    def _handle_sticker(cls, mime: Literal["sticker/tgs", "sticker/webm"],
                        file_data: BytesIO | BinaryIO) -> Blob | None:
        """
        Handle sticker processing by converting animated TGS and WEBM formats to PNG.

        Args:
            mime (Literal["sticker/tgs", "sticker/webm"]): MIME type indicating sticker format.
            file_data (BytesIO | BinaryIO): Raw sticker file data.

        Returns:
            Blob | None: A Blob object containing the processed PNG image data, or None if unrecognized or conversion fails.
        """
        mime_methods = {
            "sticker/tgs": cls._tgs,
            "sticker/webm": cls._webm
        }

        method = mime_methods.get(mime)
        if method:
            try:
                return cls._blob("image/png", method(file_data))
            except Exception as e:
                logging.error(f"Error processing {mime} sticker: {str(e)}")
                return None

        return None

    @staticmethod
    def _animation(in_file: BytesIO | BinaryIO) -> bytes:
        """
        Process an animation file (GIF/MP4) and return its data.

        Args:
            in_file (BytesIO | BinaryIO): Input animation file in binary format.

        Returns:
            bytes: The raw animation data as bytes.
        """
        # For animations, we'll just return the raw file data
        return in_file.getvalue()

    @staticmethod
    def _get_video_cover(in_file: BytesIO | BinaryIO) -> bytes:
        """
        Extract the first frame of a video and convert it to PNG format.
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
    async def download(cls, bot: Bot, file_id: str, mime: str | None = None, message: Message | None = None) -> ImageObject | Blob | None:
        """
        Download and process media from a given Telegram file ID.
        """
        if mime and mime.startswith('video/') and message and message.video:
            if message.video.file_size > cls.MAX_VIDEO_SIZE:
                file: File = await bot.get_file(file_id)
                file_data = await bot.download_file(file.file_path)

                if not file_data:
                    if message.video.thumbnail:
                        thumb_file = await bot.get_file(message.video.thumbnail.file_id)
                        thumb_data = await bot.download_file(thumb_file.file_path)
                        return cls._blob("image/jpeg", thumb_data.getvalue())

                    return None

                return cls._blob("image/png", cls._get_video_cover(file_data))

        file: File = await bot.get_file(file_id)
        file_data = await bot.download_file(file.file_path)

        if not file_data:
            return None

        # Handle videos
        if mime and mime.startswith('video/'):
            return cls._blob(mime, file_data.getvalue())

        # Handle video notes
        if mime == "video/mp4":
            return cls._blob("video/mp4", cls._video_note(file_data))

        # Handle animations
        if mime in {"video/mp4", "video/gif"}:
            return cls._blob(mime, cls._animation(file_data))

        # Handle stickers
        if mime in {"sticker/tgs", "sticker/webm"}:
            mime: Literal["sticker/tgs", "sticker/webm"]
            return cls._handle_sticker(mime, file_data)

        if mime:
            return cls._blob(mime, file_data.getvalue())

        return Image.open(file_data)
