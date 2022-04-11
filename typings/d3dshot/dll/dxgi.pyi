"""  # noqa: Y021
This type stub file was generated by pyright.
"""

import ctypes
from typing import Any, Union

import comtypes

#


class LUID(ctypes.Structure):
    _fields_ = ...


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = ...


class DXGI_OUTPUT_DESC(ctypes.Structure):
    _fields_ = ...


class DXGI_OUTDUPL_POINTER_POSITION(ctypes.Structure):
    _fields_ = ...


class DXGI_OUTDUPL_FRAME_INFO(ctypes.Structure):
    _fields_ = ...


class DXGI_MAPPED_RECT(ctypes.Structure):
    _fields_ = ...


class IDXGIObject(comtypes.IUnknown):
    _iid_ = ...
    _methods_ = ...


class IDXGIDeviceSubObject(IDXGIObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIResource(IDXGIDeviceSubObject):
    _iid_ = ...
    _methods_ = ...


class IDXGISurface(IDXGIDeviceSubObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIOutputDuplication(IDXGIObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIOutput(IDXGIObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIOutput1(IDXGIOutput):
    _iid_ = ...
    _methods_ = ...


class IDXGIAdapter(IDXGIObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIAdapter1(IDXGIAdapter):
    _iid_ = ...
    _methods_ = ...


class IDXGIFactory(IDXGIObject):
    _iid_ = ...
    _methods_ = ...


class IDXGIFactory1(IDXGIFactory):
    _iid_ = ...
    _methods_ = ...


def initialize_dxgi_factory() -> ctypes.pointer:
    ...


def discover_dxgi_adapters(dxgi_factory) -> list:
    ...


def describe_dxgi_adapter(dxgi_adapter) -> Any:
    ...


def discover_dxgi_outputs(dxgi_adapter) -> list:
    ...


def describe_dxgi_output(dxgi_output) -> dict[str, Union[Any, dict[str, Any], tuple[Any, Any], int, bool]]:
    ...


def initialize_dxgi_output_duplication(dxgi_output, d3d_device) -> ctypes.pointer:
    ...


def get_dxgi_output_duplication_frame(
        dxgi_output_duplication,
        d3d_device,
        process_func=...,
        width=...,
        height=...,
        region=...,
        rotation=...) -> None:
    ...
