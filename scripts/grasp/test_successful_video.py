import unittest

from successful_video import SuccessfulVideoRecorder


class FakeRecorder:
    def __init__(self):
        self.steps = 0
        self.encodes = 0

    def step(self):
        self.steps += 1

    def close(self):
        self.encodes += 1


class SuccessfulVideoTests(unittest.TestCase):
    def test_failed_or_interrupted_trial_never_encodes(self):
        inner = FakeRecorder()
        gate = SuccessfulVideoRecorder(inner, lambda: False)
        gate.step()
        gate.close()
        self.assertEqual(inner.steps, 1)
        self.assertEqual(inner.encodes, 0)
        self.assertIsNone(gate._recorder)

    def test_only_final_result_controls_encoding(self):
        result = {"success": False}
        inner = FakeRecorder()
        gate = SuccessfulVideoRecorder(inner, lambda: result["success"])
        gate.step()
        self.assertEqual(inner.encodes, 0)
        result["success"] = True
        gate.close()
        gate.close()
        self.assertEqual(inner.encodes, 1)


if __name__ == "__main__":
    unittest.main()
