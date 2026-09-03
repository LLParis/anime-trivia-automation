from __future__ import annotations

import ctypes
import unittest

from anime_trivia_automation.windows_input import (
    BatchedWindowsInput,
    KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE,
)


class RecordingSendInput:
    def __init__(self, *, accepted: int | None = None) -> None:
        self.accepted = accepted
        self.calls = []

    def __call__(self, count, inputs, structure_size):
        snapshot = [
            (
                int(inputs[index].type),
                int(inputs[index].ki.wVk),
                int(inputs[index].ki.wScan),
                int(inputs[index].ki.dwFlags),
            )
            for index in range(count)
        ]
        self.calls.append((count, structure_size, snapshot))
        return count if self.accepted is None else self.accepted


class BatchedWindowsInputTests(unittest.TestCase):
    def test_unicode_text_is_one_batch_with_utf16_down_up_pairs(self) -> None:
        sender = RecordingSendInput()
        native = BatchedWindowsInput(send_input=sender)

        sent = native.send_text("Aé😀")

        self.assertEqual(sent, 8)
        self.assertEqual(len(sender.calls), 1)
        count, structure_size, events = sender.calls[0]
        self.assertEqual(count, 8)
        self.assertGreaterEqual(structure_size, ctypes.sizeof(ctypes.c_void_p) * 4)
        self.assertEqual(
            [(scan, flags) for _kind, _vk, scan, flags in events],
            [
                (0x0041, KEYEVENTF_UNICODE),
                (0x0041, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
                (0x00E9, KEYEVENTF_UNICODE),
                (0x00E9, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
                (0xD83D, KEYEVENTF_UNICODE),
                (0xD83D, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
                (0xDE00, KEYEVENTF_UNICODE),
                (0xDE00, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
            ],
        )
        self.assertTrue(all(kind == 1 and vk == 0 for kind, vk, _s, _f in events))

    def test_partial_native_acceptance_fails_closed(self) -> None:
        sender = RecordingSendInput(accepted=1)
        native = BatchedWindowsInput(send_input=sender, last_error=lambda: 5)

        with self.assertRaisesRegex(OSError, "accepted 1 of 4 events"):
            native.send_text("OK")

        self.assertEqual(len(sender.calls), 1)

    def test_empty_text_is_a_noop_and_nul_is_rejected(self) -> None:
        sender = RecordingSendInput()
        native = BatchedWindowsInput(send_input=sender)

        self.assertEqual(native.send_text(""), 0)
        self.assertEqual(sender.calls, [])
        with self.assertRaisesRegex(ValueError, "NUL"):
            native.send_text("bad\x00text")


if __name__ == "__main__":
    unittest.main()
