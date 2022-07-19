""" # noqa Y021
This type stub file was partially generated by pyright.
"""
import sys

from PIL.Image import Image

#


def enum_cb(hwnd: int, results: list[tuple[int, str, str]]) -> None:
    ...


firefox: tuple[int, str] = ...
hwnd: int = ...
bbox: tuple[int, int, int, int] = ...
img: Image = ...
z1: list[str] = ...
z2: list[str] = ...
z3: str = ...

if sys.platform == "darwin":
    from pywinctl._pywinctl_macos import MacOSWindow
    my: MacOSWindow = ...
elif sys.platform == "win32":
    from pywinctl._pywinctl_win import Win32Window
    my: Win32Window = ...
elif sys.platform == "linux":
    from pywinctl._pywinctl_linux import LinuxWindow
    my: LinuxWindow = ...
x3: int = ...
y3: int = ...
p: Image = ...
im: Image = ...
im_crop: Image = ...
