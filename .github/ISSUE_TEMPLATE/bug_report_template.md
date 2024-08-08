---
name: Bug Report
about: Submit a Casanovo Bug Report
labels: bug
---

## Describe the Issue
A clear and concise description of what the issue/bug is.

## Steps To Reproduce
Steps to reproduce the incorrect behavior.

## Expected Behavior
A clear and concise description of what you expected to happen.

## Terminal Output (If Applicable)
Provide any applicable console output in between the tick marks below.

```

```

## Environment:
- OS: [e.g. Windows 11, Windows 10, macOS 14, Ubuntu 24.04]
- Casanovo Version: [e.g. 4.2.1]
- Hardware Used (CPU or GPU, if GPU also GPU model and CUDA version): [e.g. GPU: NVIDIA GeForce RTX 2070, CUDA Version: 12.5]

### Checking GPU Version

The GPU model can be checked by typing `nvidia-smi` into a terminal/console window.
An example of how to use this command is shown below.
In this case, the CUDA version is 12.5 and the GPU model is GeForce RTX 2070.


```
(casanovo_env) C:\Users\<user>\OneDrive\Documents\casanovo>nvidia-smi
Fri Aug  2 12:34:57 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.99                 Driver Version: 555.99         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2070 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   60C    P8             16W /   90W |    1059MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Additional Context
Add any other context about the problem here.

## Attach Files
Please attach all input files used and the full Casanovo log file.
