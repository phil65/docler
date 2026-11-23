"""Data models for document representation."""

from __future__ import annotations

import base64
from io import BytesIO
import mimetypes
from typing import TYPE_CHECKING, Any

from pydantic import Base64Str, Field
from schemez import MimeType, Schema
import upath
import upathtools


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


class Image(Schema):
    """Represents an image within a document."""

    id: str
    """Internal reference id used in markdown content."""

    content: bytes | Base64Str = Field(repr=False)
    """Raw image bytes or base64 encoded string."""

    mime_type: MimeType
    """MIME type of the image (e.g. 'image/jpeg', 'image/png')."""

    filename: str | None = None
    """Optional original filename of the image."""

    description: str | None = None
    """Description of the image."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the image."""

    def to_base64(self) -> str:
        """Convert image content to base64 string.

        Returns:
            Base64 encoded string of the image content.
        """
        if isinstance(self.content, bytes):
            return base64.b64encode(self.content).decode()
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        if isinstance(self.content, str) and self.content.startswith("data:"):
            return self.content.split(",", 1)[1]
        # Already a base64 string
        return self.content

    def to_base64_url(self) -> str:
        """Convert image content to base64 data URL.

        Args:
            data: Raw bytes or base64 string of image data
            mime_type: MIME type of the image

        Returns:
            Data URL format of the image for embedding in HTML/Markdown
        """
        b64_content = self.to_base64()
        return f"data:{self.mime_type};base64,{b64_content}"

    @classmethod
    async def from_file(
        cls,
        file_path: JoinablePathLike,
        image_id: str | None = None,
        description: str | None = None,
    ) -> Image:
        """Create an Image instance from a file.

        Args:
            file_path: Path to the image file
            image_id: Optional ID for the image (defaults to filename without extension)
            description: Optional description of the image

        Returns:
            Image instance with content loaded from the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        path = upath.UPath(file_path)
        if not path.exists():
            msg = f"Image file not found: {file_path}"
            raise FileNotFoundError(msg)

        mime_type, _ = mimetypes.guess_type(str(path))
        if image_id is None:
            image_id = path.stem

        content = await upathtools.read_path(path, mode="rb")
        filename = path.name
        file_stats = path.stat()
        metadata = {
            "size_bytes": file_stats.st_size,
            "created_time": file_stats.st_ctime,
            "modified_time": file_stats.st_mtime,
            "source_path": str(path),
        }

        return cls(
            id=image_id,
            content=content,
            mime_type=mime_type or "image/jpeg",
            filename=filename,
            description=description,
            metadata=metadata,
        )

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """Get the width and height of the image.

        Returns:
            A tuple of (width, height) if dimensions can be determined, None otherwise
        """
        try:
            from PIL import Image as PILImage

            if isinstance(self.content, str):
                # Handle data URLs
                if self.content.startswith("data:"):
                    # Extract the base64 part after the comma
                    base64_data = self.content.split(",", 1)[1]
                    image_data = base64.b64decode(base64_data)
                else:
                    # Regular base64 string
                    image_data = base64.b64decode(self.content)
            else:
                image_data = self.content

            # Open the image and get dimensions
            with PILImage.open(BytesIO(image_data)) as img:
                return (img.width, img.height)
        except (ImportError, Exception):  # noqa: BLE001
            return None
