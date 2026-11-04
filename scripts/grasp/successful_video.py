"""Gate Isaac Lab's buffered video encoder on the final trial result.

Configure the inner recorder with video_length=sys.maxsize and no recurrence so
that it cannot flush a clip before the trial has been evaluated.
"""


class SuccessfulVideoRecorder:
    def __init__(self, recorder, is_successful):
        self._recorder = recorder
        self._is_successful = is_successful

    def step(self):
        if self._recorder is not None:
            self._recorder.step()

    def close(self):
        recorder, self._recorder = self._recorder, None
        if recorder is not None and self._is_successful():
            recorder.close()
        # An unsuccessful trial drops its frame buffer without encoding a file.


def gate_video_recorders(env, is_successful):
    base = env.unwrapped
    base.video_recorders = [SuccessfulVideoRecorder(recorder, is_successful)
                            for recorder in base.video_recorders]
