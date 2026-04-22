import os
import tempfile
import unittest

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin


class _Dummy(SaveLoadMixin):
    def __init__(self, value=None):
        super().__init__()
        self.value = value


class TestSaveLoadMixin(unittest.TestCase):

    def test_save_and_load_roundtrip_default_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "model")
            _Dummy(value={"k": [1, 2, 3]}).save(directory)

            self.assertTrue(os.path.isfile(os.path.join(directory, "_Dummy.pkl")))

            loaded = _Dummy()
            loaded.load(directory)
            self.assertEqual(loaded.value, {"k": [1, 2, 3]})

    def test_save_and_load_with_custom_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "model")
            _Dummy(value=42).save(directory, file_name="custom.pkl")

            self.assertTrue(os.path.isfile(os.path.join(directory, "custom.pkl")))

            loaded = _Dummy()
            loaded.load(directory, file_name="custom.pkl")
            self.assertEqual(loaded.value, 42)

    def test_save_without_overwrite_raises_on_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "model")
            _Dummy(value=1).save(directory)

            with self.assertRaises(FileExistsError):
                _Dummy(value=2).save(directory)

    def test_save_with_overwrite_replaces_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "model")
            _Dummy(value="first").save(directory)
            _Dummy(value="second").save(directory, overwrite=True)

            loaded = _Dummy()
            loaded.load(directory)
            self.assertEqual(loaded.value, "second")

    def test_save_creates_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "nested", "dir", "model")
            self.assertFalse(os.path.exists(directory))
            _Dummy(value=1).save(directory)
            self.assertTrue(os.path.isdir(directory))
