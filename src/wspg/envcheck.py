"""M0 environment check: torch >= 2.7, CUDA 12.8+, GPU available, Blackwell (12, 0) kernels work."""
import sys

import torch


def main() -> int:
    ver = torch.__version__
    cuda = torch.version.cuda
    avail = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if avail else None
    print(ver, cuda, avail, cap)
    ok = True
    major, minor = (int(x) for x in ver.split("+")[0].split(".")[:2])
    if (major, minor) < (2, 7):
        print("FAIL: torch < 2.7"); ok = False
    if cuda is None or tuple(int(x) for x in cuda.split(".")[:2]) < (12, 8):
        print("FAIL: CUDA build < 12.8"); ok = False
    if not avail:
        print("FAIL: CUDA not available"); ok = False
    else:
        if cap != (12, 0):
            print(f"WARN: expected capability (12, 0), got {cap}")
        x = torch.randn(8, 4, 500, device="cuda")
        w = torch.randn(16, 4, 19, device="cuda")
        y = torch.nn.functional.conv1d(x, w)
        torch.cuda.synchronize()
        print("GPU conv1d ok:", tuple(y.shape), torch.cuda.get_device_name())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
