"""Unstructured file reader.

A parser for unstructured text files using Unstructured.io.
Supports .txt, .docx, .pptx, .jpg, .png, .eml, .html, and .pdf documents.

To use .doc and .xls parser, install

sudo apt-get install -y libmagic-dev poppler-utils libreoffice
pip install xlrd

On macOS, if file-type detection fails for extensionless or unusual files, install libmagic:

    brew install libmagic

"""
import mimetypes
from urllib.error import HTTPError
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader

from kotaemon.base import Document


def _content_type_for_partition(file_path: str) -> Optional[str]:
    """Return a MIME type Unstructured can map to FileType without libmagic.

    Unstructured runs libmagic (strategy 2) before extension-based detection (strategy 3).
    Passing ``content_type`` satisfies strategy 1 and avoids importing the native libmagic
    library when the extension (or stdlib mimetypes guess) is enough.
    """
    from unstructured.file_utils.model import FileType

    path = Path(file_path)
    ext = path.suffix.lower()
    ft = FileType.from_extension(ext)
    if ft is not None:
        return ft.mime_type
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and FileType.from_mime_type(guessed):
        return guessed
    return None


def _is_nltk_download_error(exc: Exception) -> bool:
    """Return True when unstructured fails while downloading nltk assets."""
    if isinstance(exc, HTTPError) and getattr(exc, "code", None) in (401, 403, 404):
        return True
    message = str(exc).lower()
    return "nltk" in message and ("http error" in message or "forbidden" in message)


def _join_docs_to_single_document(
    docs: List[Document], file: Path, extra_info: Optional[Dict]
) -> List[Document]:
    """Normalize fallback multi-doc output to the default single-doc contract."""
    file_name = Path(file).name
    file_path = str(Path(file).resolve())
    metadata = {"file_name": file_name, "file_path": file_path}
    if extra_info is not None:
        metadata.update(extra_info)
    text = "\n\n".join(doc.text for doc in docs if doc.text)
    return [Document(text=text, metadata=metadata)]


class UnstructuredReader(BaseReader):
    """General unstructured text reader for a variety of files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args)  # not passing kwargs to parent bc it cannot accept it

        self.api = False  # we default to local
        if "url" in kwargs:
            self.server_url = str(kwargs["url"])
            self.api = True  # is url was set, switch to api
        else:
            self.server_url = "http://localhost:8000"

        if "api" in kwargs:
            self.api = kwargs["api"]

        self.api_key = ""
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]

    """ Loads data using Unstructured.io

        Depending on the construction if url is set or api = True
        it'll parse file using API call, else parse it locally
        additional_metadata is extended by the returned metadata if
        split_documents is True

        Returns list of documents
    """

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        split_documents: Optional[bool] = False,
        **kwargs,
    ) -> List[Document]:
        """If api is set, parse through api"""
        file_path_str = str(file)
        if self.api:
            from unstructured.partition.api import partition_via_api

            elements = partition_via_api(
                filename=file_path_str,
                api_key=self.api_key,
                api_url=self.server_url + "/general/v0/general",
            )
        else:
            """Parse file locally"""
            from unstructured.partition.auto import partition

            partition_kwargs: Dict[str, Any] = {"filename": file_path_str}
            ct = _content_type_for_partition(file_path_str)
            if ct:
                partition_kwargs["content_type"] = ct
            try:
                elements = partition(**partition_kwargs)
            except ImportError as e:
                err = str(e).lower()
                if "libmagic" in err:
                    raise ImportError(
                        "File partitioning needs libmagic when the file type cannot be inferred "
                        "from the name. Install it (e.g. macOS: brew install libmagic; "
                        "Debian/Ubuntu: libmagic-dev) or use a path with a known extension."
                    ) from e
                raise
            except Exception as e:
                file_ext = Path(file_path_str).suffix.lower()
                if file_ext == ".docx" and _is_nltk_download_error(e):
                    from .docx_loader import DocxReader

                    fallback_docs = DocxReader().load_data(
                        Path(file_path_str), extra_info=extra_info
                    )
                    if split_documents:
                        return fallback_docs
                    return _join_docs_to_single_document(
                        fallback_docs, Path(file_path_str), extra_info
                    )
                raise

        """ Process elements """
        docs = []
        file_name = Path(file).name
        file_path = str(Path(file).resolve())
        if split_documents:
            for node in elements:
                metadata = {"file_name": file_name, "file_path": file_path}
                if hasattr(node, "metadata"):
                    """Load metadata fields"""
                    for field, val in vars(node.metadata).items():
                        if field == "_known_field_names":
                            continue
                        # removing coordinates because it does not serialize
                        # and dont want to bother with it
                        if field == "coordinates":
                            continue
                        # removing bc it might cause interference
                        if field == "parent_id":
                            continue
                        metadata[field] = val

                if extra_info is not None:
                    metadata.update(extra_info)

                metadata["file_name"] = file_name
                docs.append(Document(text=node.text, metadata=metadata))

        else:
            text_chunks = [" ".join(str(el).split()) for el in elements]
            metadata = {"file_name": file_name, "file_path": file_path}

            if extra_info is not None:
                metadata.update(extra_info)

            # Create a single document by joining all the texts
            docs.append(Document(text="\n\n".join(text_chunks), metadata=metadata))

        return docs
