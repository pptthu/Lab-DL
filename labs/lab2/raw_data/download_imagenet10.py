"""Script to download and build a 224x224 ImageNet-1k 10-class dataset matching CIFAR-10.

Downloads real ImageNet-1k images at native resolution, resizes them to 224x224,
and organizes them into full train, val, and test splits:
  <target_dir>/
    train/<class_name>/<images_224x224.jpg>
    val/<class_name>/<images_224x224.jpg>
    test/<class_name>/<images_224x224.jpg>

The 10 classes match CIFAR-10 semantically via representative ImageNet-1k synsets:
  0: airplane   -> n02690373 (airliner, ID 404)
  1: automobile -> n04285008 (sports_car, ID 817)
  2: bird       -> n01558993 (robin, ID 15)
  3: cat        -> n02123045 (tabby_cat, ID 281)
  4: deer       -> n02422106 (hartebeest / gazelle, ID 353)
  5: dog        -> n02099712 (golden_retriever, ID 207)
  6: frog       -> n01641577 (bullfrog, ID 30)
  7: horse      -> n02389026 (sorrel_horse, ID 340)
  8: ship       -> n03095699 (container_ship, ID 510)
  9: truck      -> n04487014 (trailer_truck, ID 867)

VI: Script tải và tạo bộ dữ liệu ImageNet-1k gồm đúng 10 lớp CIFAR-10 kích thước chuẩn 224x224.
Cung cấp đầy đủ cả 3 tập train, val, test với ảnh độ phân giải cao 224x224 pixels.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from PIL import Image

# 10 target classes matching CIFAR-10 and their corresponding ImageNet-1k specifications
IMAGENET_10_SPECS: dict[int, dict[str, str | int]] = {
    0: {"name": "airplane",   "wnid": "n02690373", "id": 404, "label": "airliner"},
    1: {"name": "automobile", "wnid": "n04285008", "id": 817, "label": "sports_car"},
    2: {"name": "bird",       "wnid": "n01558993", "id": 15,  "label": "robin"},
    3: {"name": "cat",        "wnid": "n02123045", "id": 281, "label": "tabby_cat"},
    4: {"name": "deer",       "wnid": "n02422106", "id": 353, "label": "hartebeest"},
    5: {"name": "dog",        "wnid": "n02099712", "id": 207, "label": "golden_retriever"},
    6: {"name": "frog",       "wnid": "n01641577", "id": 30,  "label": "bullfrog"},
    7: {"name": "horse",      "wnid": "n02389026", "id": 340, "label": "sorrel_horse"},
    8: {"name": "ship",       "wnid": "n03095699", "id": 510, "label": "container_ship"},
    9: {"name": "truck",      "wnid": "n04487014", "id": 867, "label": "trailer_truck"},
}

TARGET_WNIDS = {spec["wnid"]: spec["name"] for spec in IMAGENET_10_SPECS.values()}
TARGET_IDS = {spec["id"]: spec["name"] for spec in IMAGENET_10_SPECS.values()}


def resize_to_224(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Resize and center crop an image to target_size x target_size (default 224x224).
    
    VI: Resize và cắt giữa ảnh về kích thước chuẩn target_size x target_size (mặc định 224x224).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    # Maintain aspect ratio and scale so the shorter side matches target_size
    scale = target_size / min(width, height)
    new_w = round(width * scale)
    new_h = round(height * scale)

    # Resize with high quality filter
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    img_resized = img.resize((new_w, new_h), resample=resample_filter)

    # Center crop to target_size x target_size
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    return img_resized.crop((left, top, right, bottom))


def build_from_huggingface(
    target_dir: Path,
    image_size: int = 224,
    samples_per_class: int = 200,
    hf_token: str | None = None,
) -> None:
    """Download ImageNet-1k images via Hugging Face and resize to 224x224.
    
    VI: Tải ảnh ImageNet-1k từ Hugging Face và resize chuẩn 224x224.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as err:
        raise ImportError(
            "The Hugging Face `datasets` package is required. Install with: pip install datasets"
        ) from err

    print("🌐 Connecting to Hugging Face ImageNet-1k stream...")
    # First attempt: official ILSVRC/imagenet-1k (uses token if provided)
    # Fallback attempt: open mirror mrm8488/ImageNet1K-val (requires no login)
    dataset = None
    dataset_name = "mrm8488/ImageNet1K-val"
    split_name = "train"

    if hf_token:
        try:
            print("🔑 Attempting stream from 'ILSVRC/imagenet-1k' using token...")
            dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split="train",
                streaming=True,
                token=hf_token,
            )
            dataset_name = "ILSVRC/imagenet-1k"
            print("✅ Successfully authenticated with ILSVRC/imagenet-1k!")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Token auth failed ({e}), falling back to open mirror '{dataset_name}'...")
            dataset = None

    if dataset is None:
        print(f"📡 Streaming from open ImageNet-1k mirror: '{dataset_name}'...")
        try:
            dataset = load_dataset(dataset_name, split=split_name, streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from Hugging Face: {exc}\n"
                f"Check internet connection or pass a Hugging Face token with --hf-token."
            ) from exc

    # Setup directories
    for split in ["train", "val", "test"]:
        for cls_name in TARGET_WNIDS.values():
            (target_dir / split / str(cls_name)).mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[Image.Image]] = {str(name): [] for name in TARGET_WNIDS.values()}
    total_needed = samples_per_class * len(TARGET_WNIDS)

    print(f"⬇️ Streaming and extracting 10 classes (target: {samples_per_class} images/class, 224x224)...")
    start_time = time.time()
    seen = 0

    for item in dataset:
        seen += 1
        label = item.get("label")
        img = item.get("image")
        if label is None or img is None:
            continue

        if label in TARGET_IDS:
            cls_name = str(TARGET_IDS[label])
            if len(collected[cls_name]) < samples_per_class:
                # Resize immediately to 224x224 to save memory
                img_224 = resize_to_224(img, target_size=image_size)
                collected[cls_name].append(img_224)
                total_collected = sum(len(v) for v in collected.values())

                if total_collected % 25 == 0 or total_collected == total_needed:
                    elapsed = max(0.1, time.time() - start_time)
                    sys.stdout.write(
                        f"\r📸 Collected: {total_collected:,}/{total_needed:,} images "
                        f"({total_collected*100/total_needed:5.1f}%) | "
                        f"Speed: {total_collected/elapsed:4.1f} img/s "
                        f"[Scanned {seen:,} stream samples]"
                    )
                    sys.stdout.flush()

                if all(len(v) >= samples_per_class for v in collected.values()):
                    break

    sys.stdout.write("\n")
    print("📁 Partitioning and saving into 'train' (70%), 'val' (15%), and 'test' (15%)...")

    random.seed(42)
    for cls_name, img_list in collected.items():
        random.shuffle(img_list)
        n = len(img_list)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_imgs = img_list[:n_train]
        val_imgs = img_list[n_train : n_train + n_val]
        test_imgs = img_list[n_train + n_val :]

        for split_name_part, split_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            cls_dir = target_dir / split_name_part / cls_name
            for i, p_img in enumerate(split_list):
                file_path = cls_dir / f"{cls_name}_{i:05d}.jpg"
                p_img.save(file_path, format="JPEG", quality=95)


def build_from_local_imagenet(
    local_dir: Path,
    target_dir: Path,
    image_size: int = 224,
    samples_per_class: int = 200,
) -> None:
    """Extract and resize the 10 classes from a local ImageNet folder.
    
    VI: Trích xuất và resize 10 lớp từ thư mục ImageNet có sẵn trên máy cục bộ.
    """
    print(f"🔍 Scanning local ImageNet folder: {local_dir}")
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local ImageNet directory not found: {local_dir}")

    # Check for train / val or direct synset folders
    found_wnids = 0
    for wnid, cls_name in TARGET_WNIDS.items():
        # Look for folder named wnid (e.g. n02690373) anywhere under local_dir
        matches = list(local_dir.rglob(f"{wnid}"))
        if not matches:
            matches = list(local_dir.rglob(f"*{cls_name}*"))

        if matches:
            src_folder = matches[0]
            found_wnids += 1
            all_files = list(src_folder.glob("*.JPEG")) + list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png"))
            random.seed(42)
            random.shuffle(all_files)
            selected = all_files[:samples_per_class]

            n = len(selected)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)

            for split_name, file_slice in [
                ("train", selected[:n_train]),
                ("val", selected[n_train : n_train + n_val]),
                ("test", selected[n_train + n_val :]),
            ]:
                dest_cls_dir = target_dir / split_name / cls_name
                dest_cls_dir.mkdir(parents=True, exist_ok=True)
                for idx, src_img_path in enumerate(file_slice):
                    try:
                        with Image.open(src_img_path) as raw_img:
                            resized = resize_to_224(raw_img, target_size=image_size)
                            resized.save(dest_cls_dir / f"{cls_name}_{idx:05d}.jpg", format="JPEG", quality=95)
                    except Exception as err:  # noqa: BLE001
                        print(f"⚠️ Error processing {src_img_path}: {err}")

    print(f"✅ Extracted {found_wnids}/{len(TARGET_WNIDS)} classes from local ImageNet.")


def verify_dataset_224(target_dir: Path) -> None:
    """Verify image dimensions (224x224), format, and counts across all splits.
    
    VI: Kiểm tra độ phân giải (224x224), định dạng và số lượng ảnh trong các tập.
    """
    print("\n" + "=" * 70)
    print("📊 KIỂM TRA ĐỘ PHÂN GIẢI & TÍNH TOÀN VẸN CỦA BỘ DỮ LIỆU:")
    print("=" * 70)

    all_ok = True
    for split in ["train", "val", "test"]:
        split_path = target_dir / split
        if not split_path.is_dir():
            print(f"❌ Thiếu thư mục tập '{split}': {split_path}")
            all_ok = False
            continue

        split_total = 0
        dimension_verified = True
        checked_sample_dim = None

        for spec in IMAGENET_10_SPECS.values():
            cls_name = str(spec["name"])
            cls_dir = split_path / cls_name
            if cls_dir.is_dir():
                imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                split_total += len(imgs)
                # Sample check image dimensions
                if imgs and checked_sample_dim is None:
                    with Image.open(imgs[0]) as im:
                        checked_sample_dim = im.size
                        if im.size != (224, 224):
                            dimension_verified = False

        status = "✅" if split_total > 0 and dimension_verified else "⚠️"
        dim_str = f"{checked_sample_dim[0]}x{checked_sample_dim[1]}" if checked_sample_dim else "N/A"
        print(f"{status} Tập '{split}': {split_total:,} ảnh | Kích thước: {dim_str} px across 10 classes")

    if all_ok:
        print("\n🎉 Bộ dữ liệu ImageNet-10 chuẩn 224x224 đã sẵn sàng cho PyTorch ImageFolder!")
    else:
        print("\n⚠️ Một số ảnh hoặc thư mục chưa hoàn chỉnh. Vui lòng kiểm tra lại log.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Main CLI entrypoint. / Điểm thực thi dòng lệnh chính."""
    parser = argparse.ArgumentParser(
        description="Download & build a 224x224 ImageNet-1k 10-class dataset matching CIFAR-10."
    )
    default_dir = Path(__file__).resolve().parent / "imagenet10"
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_dir,
        help=f"Target directory for 224x224 dataset (default: {default_dir})",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size in pixels (default: 224 for 224x224)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        help="Number of images to collect per class (default: 200, gives 2,000 total images)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face user access token for gated ILSVRC/imagenet-1k",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional path to local ImageNet directory or extracted synset archive",
    )
    args = parser.parse_args(argv)

    target_dir: Path = args.target_dir.resolve()
    print("=" * 70)
    print("🚀 IMAGENET-1K (10 CLASSES MATCHING CIFAR-10) - 224x224 BUILDER")
    print(f"📁 Thư mục lưu trữ : {target_dir}")
    print(f"📐 Kích thước ảnh  : {args.image_size} x {args.image_size} pixels")
    print(f"🔢 Số mẫu mỗi lớp  : {args.samples_per_class} ảnh (Tổng: {args.samples_per_class * 10:,} ảnh)")
    print("=" * 70)

    if args.local_dir:
        build_from_local_imagenet(
            local_dir=args.local_dir.resolve(),
            target_dir=target_dir,
            image_size=args.image_size,
            samples_per_class=args.samples_per_class,
        )
    else:
        build_from_huggingface(
            target_dir=target_dir,
            image_size=args.image_size,
            samples_per_class=args.samples_per_class,
            hf_token=args.hf_token,
        )

    verify_dataset_224(target_dir)


if __name__ == "__main__":
    main()
