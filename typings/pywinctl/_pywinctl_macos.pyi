""" # noqa Y021
This type stub file was partially generated by pyright.
"""
import sys

if sys.platform != "darwin":
    return  # pyright: ignore # noqa: F706

from typing import Literal

import AppKit
from pywinctl import BaseWindow, Point, Rect, Size

WS = ...
WAIT_ATTEMPTS = ...
WAIT_DELAY = ...
SEP = ...


def checkPermissions(activate: bool = ...) -> bool:
    ...


def getActiveWindow(app: AppKit.NSApplication = ...) -> MacOSWindow | MacOSNSWindow | None:
    ...


def getActiveWindowTitle(app: AppKit.NSApplication = ...) -> str:
    ...


def getAllWindows(app: AppKit.NSApplication = ...) -> list[MacOSWindow]:
    ...


def getAllTitles(app: AppKit.NSApplication = ...) -> list[str]:
    ...


def getWindowsWithTitle(title, app=..., condition=..., flags=...):
    ...


def getAllAppsNames() -> list[str]:
    ...


def getAppsWithName(name, condition=..., flags=...):
    ...


def getAllAppsWindowsTitles() -> dict:
    ...


def getWindowsAt(x: int, y: int, app: AppKit.NSApplication = ..., allWindows=...) -> list[MacOSWindow]:
    ...


class MacOSWindow(BaseWindow):
    def __init__(self, app: AppKit.NSRunningApplication, title: str, bounds: Rect = ...) -> None:
        ...

    def getExtraFrameSize(self, includeBorder: bool = ...) -> tuple[int, int, int, int]:
        ...

    def getClientFrame(self) -> Rect:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other) -> bool:
        ...

    def close(self, force: bool = ...) -> bool:
        ...

    def minimize(self, wait: bool = ...) -> bool:
        ...

    def maximize(self, wait: bool = ...) -> bool:
        ...

    def restore(self, wait: bool = ...) -> bool:
        ...

    def show(self, wait: bool = ...) -> bool:
        ...

    def hide(self, wait: bool = ...) -> bool:
        ...

    def activate(self, wait: bool = ...) -> bool:
        ...

    def resize(self, widthOffset: int, heightOffset: int, wait: bool = ...) -> bool:
        ...

    resizeRel = ...

    def resizeTo(self, newWidth: int, newHeight: int, wait: bool = ...) -> bool:
        ...

    def move(self, xOffset: int, yOffset: int, wait: bool = ...) -> bool:
        ...

    moveRel = ...

    def moveTo(self, newLeft: int, newTop: int, wait: bool = ...) -> bool:
        ...

    def alwaysOnTop(self, aot: bool = ...) -> bool:
        ...

    def alwaysOnBottom(self, aob: bool = ...) -> bool:
        ...

    def lowerWindow(self) -> None:
        ...

    def raiseWindow(self) -> None:
        ...

    def sendBehind(self, sb: bool = ...) -> bool:
        ...

    def getAppName(self) -> str:
        ...

    def getParent(self) -> str:
        ...

    def getChildren(self) -> list[Unknown]:
        ...

    def getHandle(self) -> str:
        ...

    def isParent(self, child: str) -> bool:
        ...

    isParentOf = ...

    def isChild(self, parent: str) -> bool:
        ...

    isChildOf = ...

    def getDisplay(self) -> Literal['']:
        ...

    @property
    def isMinimized(self) -> bool:
        ...

    @property
    def isMaximized(self) -> bool:
        ...

    @property
    def isActive(self) -> bool:
        ...

    @property
    def title(self) -> str | None:
        ...

    @property
    def updatedTitle(self) -> str:
        ...

    @property
    def visible(self) -> bool:
        ...

    isVisible = ...

    @property
    def isAlive(self) -> bool:
        ...

    class _WatchDog:

        def __init__(self, parent) -> None:
            ...

        def start(
                self,
                isAliveCB=...,
                isActiveCB=...,
                isVisibleCB=...,
                isMinimizedCB=...,
                isMaximizedCB=...,
                resizedCB=...,
                movedCB=...,
                changedTitleCB=...,
                changedDisplayCB=...,
                interval=...) -> None:
            ...

        def updateCallbacks(
                self,
                isAliveCB=...,
                isActiveCB=...,
                isVisibleCB=...,
                isMinimizedCB=...,
                isMaximizedCB=...,
                resizedCB=...,
                movedCB=...,
                changedTitleCB=...,
                changedDisplayCB=...) -> None:
            ...

        def updateInterval(self, interval=...) -> None:
            ...

        def setTryToFind(self, tryToFind: bool) -> None:
            ...

        def stop(self) -> None:
            ...

        def isAlive(self) -> bool:
            ...

    class _Menu:
        def __init__(self, parent: BaseWindow) -> None:
            ...

        def getMenu(self, addItemInfo: bool = ...) -> dict:
            ...

        def clickMenuItem(self, itemPath: list = ..., wID: int = ...) -> bool:
            ...

        def getMenuInfo(self, hSubMenu: int) -> dict:
            ...

        def getMenuItemCount(self, hSubMenu: int) -> int:
            ...

        def getMenuItemInfo(self, hSubMenu: int, wID: int) -> dict:
            ...

        def getMenuItemRect(self, hSubMenu: int, wID: int) -> Rect:
            ...


class MacOSNSWindow(BaseWindow):
    def __init__(self, app: AppKit.NSApplication, hWnd: AppKit.NSWindow) -> None:
        ...

    def getExtraFrameSize(self, includeBorder: bool = ...) -> tuple[int, int, int, int]:
        ...

    def getClientFrame(self) -> Rect:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other) -> bool:
        ...

    def close(self) -> bool:
        ...

    def minimize(self, wait: bool = ...) -> bool:
        ...

    def maximize(self, wait: bool = ...) -> bool:
        ...

    def restore(self, wait: bool = ...) -> bool:
        ...

    def show(self, wait: bool = ...) -> bool:
        ...

    def hide(self, wait: bool = ...) -> bool:
        ...

    def activate(self, wait: bool = ...) -> bool:
        ...

    def resize(self, widthOffset: int, heightOffset: int, wait: bool = ...) -> bool:
        ...

    resizeRel = ...

    def resizeTo(self, newWidth: int, newHeight: int, wait: bool = ...) -> bool:
        ...

    def move(self, xOffset: int, yOffset: int, wait: bool = ...) -> bool:
        ...

    moveRel = ...

    def moveTo(self, newLeft: int, newTop: int, wait: bool = ...) -> bool:
        ...

    def alwaysOnTop(self, aot: bool = ...) -> bool:
        ...

    def alwaysOnBottom(self, aob: bool = ...) -> bool:
        ...

    def lowerWindow(self) -> bool:
        ...

    def raiseWindow(self, sb: bool = ...) -> bool:
        ...

    def sendBehind(self, sb: bool = ...) -> bool:
        ...

    def getAppName(self) -> str:
        ...

    def getParent(self) -> int:
        ...

    def getChildren(self) -> list[int]:
        ...

    def getHandle(self) -> int:
        ...

    def isParent(self, child) -> bool:
        ...

    isParentOf = ...

    def isChild(self, parent) -> bool:
        ...

    isChildOf = ...

    def getDisplay(self) -> Literal['']:
        ...

    @property
    def isMinimized(self) -> bool:
        ...

    @property
    def isMaximized(self) -> bool:
        ...

    @property
    def isActive(self) -> bool:
        ...

    @property
    def title(self) -> str:
        ...

    @property
    def visible(self) -> bool:
        ...

    isVisible = ...

    @property
    def isAlive(self) -> bool:
        ...

    class _WatchDog:

        def __init__(self, parent) -> None:
            ...

        def start(
                self,
                isAliveCB=...,
                isActiveCB=...,
                isVisibleCB=...,
                isMinimizedCB=...,
                isMaximizedCB=...,
                resizedCB=...,
                movedCB=...,
                changedTitleCB=...,
                changedDisplayCB=...,
                interval=...) -> None:
            ...

        def updateCallbacks(
                self,
                isAliveCB=...,
                isActiveCB=...,
                isVisibleCB=...,
                isMinimizedCB=...,
                isMaximizedCB=...,
                resizedCB=...,
                movedCB=...,
                changedTitleCB=...,
                changedDisplayCB=...) -> None:
            ...

        def updateInterval(self, interval=...) -> None:
            ...

        def setTryToFind(self, tryToFind: bool) -> None:
            ...

        def stop(self) -> None:
            ...

        def isAlive(self) -> bool:
            ...


def getMousePos() -> Point:
    ...


cursor = ...


def getAllScreens() -> dict[Unknown, Unknown]:
    ...


def getScreenSize(name: str = ...) -> Size:
    ...


resolution = ...


def getWorkArea(name: str = ...) -> Rect:
    ...


def displayWindowsUnderMouse(xOffset: int = ..., yOffset: int = ...) -> None:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    ...
