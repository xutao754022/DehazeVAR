from __future__ import annotations

import random
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TrainTransform:
    def __init__(
        self,
        resize: int,
        crop: int,
        horizontal_flip: bool,
        preserve_aspect_ratio: bool = False,
    ) -> None:
        self.resize = resize
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resize_size: int | list[int]
        if self.preserve_aspect_ratio:
            resize_size = self.resize
        else:
            resize_size = [self.resize, self.resize]
        image = TF.resize(
            image,
            resize_size,
            interpolation=transforms.InterpolationMode.LANCZOS,
            antialias=True,
        )
        top, left, height, width = transforms.RandomCrop.get_params(
            image, output_size=(self.crop, self.crop)
        )
        image = TF.crop(image, top, left, height, width)
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def inference_transform(size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.LANCZOS,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.detach().float().add(1.0).mul(0.5).clamp(0.0, 1.0)


def image_files_recursive(image_directory: str | Path) -> list[Path]:
    """Return a deterministic recursive image list for an explicit domain root."""
    image_directory = Path(image_directory).expanduser().resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")
    images = sorted(
        path
        for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in {image_directory}")
    return images


def image_files_from_official_manifest(
    image_directory: str | Path,
    manifest_path: str | Path,
 ) -> list[Path]:
    """Resolve a Places-style ``<relative-path> <class-id>`` subset index."""
    image_directory = Path(image_directory).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Image manifest does not exist: {manifest_path}")

    images: list[Path] = []
    seen_relative_paths: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(
                f"Malformed manifest line {line_number} in {manifest_path}: {raw_line!r}"
            )
        official_path, class_id_text = fields
        try:
            int(class_id_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid class id on line {line_number} in {manifest_path}: "
                f"{class_id_text!r}"
            ) from error

        relative_path = PurePosixPath(official_path.lstrip("/"))
        if not relative_path.parts or any(
            part in {"", ".", ".."} for part in relative_path.parts
        ):
            raise ValueError(
                f"Unsafe dataset path on line {line_number} in {manifest_path}: "
                f"{official_path!r}"
            )
        relative_key = relative_path.as_posix()
        if relative_key in seen_relative_paths:
            raise ValueError(
                f"Duplicate dataset path on line {line_number} in {manifest_path}: "
                f"{official_path!r}"
            )
        seen_relative_paths.add(relative_key)

        image_path = image_directory.joinpath(*relative_path.parts).resolve()
        try:
            image_path.relative_to(image_directory)
        except ValueError as error:
            raise ValueError(
                f"Dataset path escapes image root on line {line_number}: "
                f"{official_path!r}"
            ) from error
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension on line {line_number}: {official_path!r}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Manifest image is missing on line {line_number}: {image_path}"
            )
        images.append(image_path)

    if not images:
        raise RuntimeError(f"No supported images listed in {manifest_path}")
    return images


class UnpairedDehazingDataset(Dataset):
    """Sample the real-hazy and clean domains independently."""

    def __init__(
        self,
        hazy_image_directory: str | Path,
        clean_image_directory: str | Path,
        resize: int = 256,
        crop: int = 256,
        horizontal_flip: bool = True,
        overfit_single_pair: bool = False,
        additional_clean_image_directory: str | Path | None = None,
        additional_clean_image_manifest: str | Path | None = None,
        additional_clean_sampling_probability: float = 0.0,
    ) -> None:
        self.hazy_image_directory = Path(hazy_image_directory).expanduser().resolve()
        self.clean_image_directory = Path(clean_image_directory).expanduser().resolve()
        self.hazy_images = image_files_recursive(self.hazy_image_directory)
        self.primary_clean_images = image_files_recursive(self.clean_image_directory)
        self.additional_clean_sampling_probability = float(
            additional_clean_sampling_probability
        )
        if not 0.0 <= self.additional_clean_sampling_probability <= 1.0:
            raise ValueError(
                "additional_clean_sampling_probability must be between zero and one"
            )
        if (additional_clean_image_directory is None) != (
            additional_clean_image_manifest is None
        ):
            raise ValueError(
                "additional_clean_image_directory and "
                "additional_clean_image_manifest must be configured together"
            )
        if additional_clean_image_directory is None:
            self.additional_clean_image_directory = None
            self.additional_clean_image_manifest = None
            self.additional_clean_images: list[Path] = []
        else:
            self.additional_clean_image_directory = (
                Path(additional_clean_image_directory).expanduser().resolve()
            )
            self.additional_clean_image_manifest = (
                Path(additional_clean_image_manifest).expanduser().resolve()
            )
            self.additional_clean_images = image_files_from_official_manifest(
                self.additional_clean_image_directory,
                self.additional_clean_image_manifest,
            )
        if self.additional_clean_sampling_probability > 0.0 and not (
            self.additional_clean_images
        ):
            raise ValueError(
                "A positive additional clean sampling probability requires "
                "a non-empty manifest"
            )
        self.clean_images = [
            *self.primary_clean_images,
            *self.additional_clean_images,
        ]
        self.transform = TrainTransform(
            resize,
            crop,
            horizontal_flip,
            preserve_aspect_ratio=True,
        )
        self.overfit_single_pair = overfit_single_pair

    def __len__(self) -> int:
        return max(len(self.hazy_images), len(self.clean_images))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.overfit_single_pair:
            hazy_path = self.hazy_images[0]
            clean_path = self.primary_clean_images[0]
            clean_source = "primary"
        else:
            hazy_path = self.hazy_images[index % len(self.hazy_images)]
            use_additional = bool(self.additional_clean_images) and (
                random.random() < self.additional_clean_sampling_probability
            )
            clean_pool = (
                self.additional_clean_images
                if use_additional
                else self.primary_clean_images
            )
            clean_path = random.choice(clean_pool)
            clean_source = "additional" if use_additional else "primary"

        with Image.open(hazy_path) as hazy_image:
            hazy = self.transform(hazy_image.convert("RGB"))
        with Image.open(clean_path) as clean_image:
            clean = self.transform(clean_image.convert("RGB"))
        return {
            "hazy": hazy,
            "clean": clean,
            "hazy_path": str(hazy_path),
            "clean_path": str(clean_path),
            "clean_source": clean_source,
        }


__all__ = [
    "IMAGE_EXTENSIONS",
    "TrainTransform",
    "UnpairedDehazingDataset",
    "denormalize",
    "image_files_from_official_manifest",
    "image_files_recursive",
    "inference_transform",
]
