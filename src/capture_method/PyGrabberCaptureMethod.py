from __future__ import annotations

from threading import Event, Thread
from typing import TYPE_CHECKING

import cv2
from pygrabber.dshow_graph import FilterGraph, StateGraph

from capture_method.CaptureMethodBase import CaptureMethodBase
from error_messages import CREATE_NEW_ISSUE_MESSAGE, exception_traceback
from utils import is_valid_image

if TYPE_CHECKING:
    from AutoSplit import AutoSplit


class PyGrabberCaptureMethod(CaptureMethodBase):
    graph: FilterGraph
    capture_thread: Thread | None
    last_captured_frame: cv2.Mat | None = None
    is_old_image = False
    is_last_capture_success = False
    stop_thread = Event()

    def __frame_grabbed(self, image: cv2.Mat):
        self.last_captured_frame = image
        self.is_old_image = False

    def __read_loop(self, autosplit: AutoSplit):
        try:
            while not self.stop_thread.is_set():
                self.is_last_capture_success = self.graph.grab_frame()
        except Exception as exception:  # pylint: disable=broad-except # We really want to catch everything here
            error = exception
            self.graph.stop()
            autosplit.show_error_signal.emit(lambda: exception_traceback(
                "AutoSplit encountered an unhandled exception while trying to grab a frame and has stopped capture. "
                + CREATE_NEW_ISSUE_MESSAGE,
                error))

    def __init__(self, autosplit: AutoSplit):
        super().__init__()
        self.graph = FilterGraph()
        self.graph.add_video_input_device(autosplit.settings_dict["capture_device_id"])
        self.graph.add_sample_grabber(self.__frame_grabbed)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.stop_thread = Event()
        self.capture_thread = Thread(target=lambda: self.__read_loop(autosplit))
        self.capture_thread.start()

    def close(self, autosplit: AutoSplit):
        self.stop_thread.set()
        if self.capture_thread:
            self.capture_thread.join()
            self.capture_thread = None
        self.graph.stop()

    def get_frame(self, autosplit: AutoSplit):
        selection = autosplit.settings_dict["capture_region"]
        if not self.check_selected_region_exists(autosplit):
            return None, False

        if not self.is_last_capture_success:
            return None, self.is_old_image

        image = self.last_captured_frame
        is_old_image = self.is_old_image
        self.is_old_image = True
        if not is_valid_image(image):
            return None, is_old_image

        # Ensure we can't go OOB of the image
        y = min(selection["y"], image.shape[0] - 1)
        x = min(selection["x"], image.shape[1] - 1)
        image = image[
            y:y + selection["height"],
            x:x + selection["width"],
        ]
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), is_old_image

    def recover_window(self, captured_window_title: str, autosplit: AutoSplit) -> bool:
        raise NotImplementedError()

    def check_selected_region_exists(self, autosplit: AutoSplit):
        return self.graph.get_state() == StateGraph.Running
