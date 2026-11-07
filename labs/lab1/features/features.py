"""Module `features` (Feature engineering (transformation + scaling)) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (biến đổi + chuẩn hóa)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
# Input / Output of this module
#   Input : images, labels, train_idx, val_idx from preprocessing (train CSV + artifacts/splits/train_val_split.json);
#           test images and labels from the test CSV
#   Output: in memory only, nothing is written to disk:
#           (mean, std) = (0.2859, 0.3530); train_loader (844 batches), val_loader (94), test_loader (157);
#           each batch = images [64,1,28,28] float32 + labels [64] int64; used by train, evaluate, visualize, experiment
# VI: Input / Output của module
#   Vào : images, labels, train_idx, val_idx từ preprocessing (CSV train + artifacts/splits/train_val_split.json);
#         ảnh và nhãn test từ CSV test
#   Ra  : chỉ trong bộ nhớ, không ghi file nào:
#         (mean, std) = (0,2859, 0,3530); train_loader (844 batch), val_loader (94), test_loader (157);
#         mỗi batch = ảnh [64,1,28,28] float32 + nhãn [64] int64; dùng bởi train, evaluate, visualize, experiment

# Type hints are not evaluated at import time (works on older Python versions too).
# VI: Không tính type hint lúc import (chạy được cả trên Python cũ).
from __future__ import annotations

# numpy: image arrays; torch: tensors; Dataset/DataLoader: PyTorch data API.
# VI: numpy: mảng ảnh; torch: tensor; Dataset/DataLoader: API dữ liệu của PyTorch.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Largest 8-bit pixel value; a float so every division gives a float.
# VI: Giá trị pixel 8 bit lớn nhất; để dạng số thực để phép chia luôn ra số thực.
PIXEL_MAX = 255.0


def compute_mean_std(train_images) -> tuple:
    """Compute normalization statistics from the TRAIN split only.

    Input : train_images: uint8 array [N,28,28] (train indices only)
    Output: (mean, std) as floats on the 0-1 pixel scale

    VI: Tính mean/std để chuẩn hóa, CHỈ dùng tập train.
    Vào : train_images: mảng uint8 [N,28,28] (chỉ chỉ số train)
    Ra  : (mean, std) dạng số thực trên thang pixel 0-1
    """
    # Accept any array-like (list, numpy array).
    # VI: Nhận mọi kiểu giống mảng (list, mảng numpy).
    images = np.asarray(train_images)
    # One mean/std for the whole split (grayscale = 1 channel). Dividing by 255
    # afterwards gives the same result as scaling every pixel to 0-1 first.
    # VI: Một mean/std cho cả tập (ảnh xám = 1 kênh). Chia 255 sau cùng cho kết quả
    # giống như đưa từng pixel về 0-1 trước rồi mới tính.
    # float64 accumulator: summing ~42 million uint8 pixels in float32 would lose precision.
    # VI: Cộng dồn bằng float64: cộng khoảng 42 triệu pixel bằng float32 sẽ mất độ chính xác.
    mean = float(images.mean(dtype=np.float64) / PIXEL_MAX)
    std = float(images.std(dtype=np.float64) / PIXEL_MAX)
    # Plain Python floats; on the official train split: 0.2859, 0.3530.
    # VI: Số thực Python thường; trên tập train chính thức: 0,2859, 0,3530.
    return mean, std


def build_transform(mean: float, std: float):
    """Build the transform applied to every image.

    Input : mean, std
    Output: callable: uint8 [28,28] -> float32 tensor [1,28,28], normalized

    VI: Tạo phép biến đổi áp dụng cho mọi ảnh.
    Vào : mean, std
    Ra  : hàm: uint8 [28,28] -> tensor float32 [1,28,28] đã chuẩn hóa
    """
    # Dividing by 0 (or a negative std) would give inf/NaN pixels.
    # VI: Chia cho 0 (hoặc std âm) sẽ ra pixel inf/NaN.
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")

    # Inner function (closure): remembers mean and std without writing a class.
    # VI: Hàm lồng (closure): nhớ mean và std mà không cần viết lớp.
    def transform(image) -> torch.Tensor:
        # uint8 [28,28] in 0-255 -> float32 [28,28] in 0-1 -> add channel dim -> normalize.
        # VI: uint8 [28,28] thang 0-255 -> float32 [28,28] thang 0-1 -> thêm chiều kênh -> chuẩn hóa.
        # float32 = the dtype of the model weights.
        # VI: float32 = kiểu dữ liệu của trọng số model.
        x = torch.as_tensor(np.asarray(image), dtype=torch.float32) / PIXEL_MAX
        # unsqueeze(0): [28,28] -> [1,28,28]; then (x - mean) / std gives mean 0, std 1 on train.
        # VI: unsqueeze(0): [28,28] -> [1,28,28]; rồi (x - mean) / std cho mean 0, std 1 trên train.
        return (x.unsqueeze(0) - mean) / std

    # The same function is used for train, validation and test.
    # VI: Cùng một hàm dùng cho train, validation và test.
    return transform


class FashionDataset(Dataset):
    """Dataset wrapper (__init__, __len__, __getitem__).

    Input : __init__(images, labels, indices, transform)
    Output: __getitem__(i) -> (tensor [1,28,28], int label); __len__() -> number of samples

    VI: Lớp Dataset (__init__, __len__, __getitem__).
    Vào : __init__(images, labels, indices, transform)
    Ra  : __getitem__(i) -> (tensor [1,28,28], nhãn int); __len__() -> số mẫu
    """

    def __init__(self, images, labels, indices=None, transform=None) -> None:
        # `indices` selects the rows of this split (train or val). None = use every
        # row (test set). The image array is shared, not copied.
        # VI: `indices` chọn các dòng thuộc split này (train hoặc val). None = dùng hết
        # (tập test). Mảng ảnh được dùng chung, không sao chép.
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)
        # Applied lazily, one image at a time, when the DataLoader asks for it.
        # VI: Áp dụng khi cần, từng ảnh một, lúc DataLoader lấy ra.
        self.transform = transform

    def __len__(self) -> int:
        # The DataLoader needs the number of samples (54,000 / 6,000 / 10,000).
        # VI: DataLoader cần biết số mẫu (54.000 / 6.000 / 10.000).
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple:
        # Position i in this split -> row number in the shared array.
        # VI: Vị trí i trong tập này -> số dòng trong mảng dùng chung.
        row = self.indices[i]
        image = self.images[row]
        # Normalize only when a transform is given (tests may use raw arrays).
        # VI: Chỉ chuẩn hóa khi có transform (test có thể dùng mảng thô).
        if self.transform is not None:
            image = self.transform(image)
        # int label; the default collate stacks a batch of labels into an int64 tensor.
        # VI: Nhãn int; collate mặc định gộp nhãn của batch thành tensor int64.
        return image, int(self.labels[row])


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple:
    """Create DataLoaders (shuffle the training loader only).

    Input : three Dataset objects; batch_size
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader (chỉ shuffle tập train).
    Vào : ba đối tượng Dataset; batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """
    # Shuffle train only: a new batch order every epoch helps SGD; val/test order
    # does not change the metrics, so keep it fixed.
    # VI: Chỉ shuffle train: thứ tự batch mới mỗi epoch giúp SGD; thứ tự val/test
    # không đổi kết quả đo nên giữ cố định.
    # drop_last=False (default): 844 / 94 / 157 batches, the last ones are smaller; no image is dropped.
    # VI: drop_last=False (mặc định): 844 / 94 / 157 batch, batch cuối nhỏ hơn; không bỏ ảnh nào.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
