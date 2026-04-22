import unittest
import warnings

from energy_fault_detector.utils._deprecations import deprecate_kwargs


# The decorator expects the wrapped function to still accept both the deprecated
# and the replacement parameter names in its signature — it only translates
# call-time kwargs. Tests below follow that real-world usage pattern.


def _make_wrapped(prefer="old"):
    @deprecate_kwargs({"old_name": "new_name"}, prefer=prefer)
    def fn(new_name=None, *, old_name=None):
        return new_name
    return fn


class TestDeprecateKwargs(unittest.TestCase):

    def test_old_kwarg_is_forwarded_to_new(self):
        fn = _make_wrapped()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(old_name=42)

        self.assertEqual(result, 42)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_new_kwarg_alone_emits_no_warning(self):
        fn = _make_wrapped()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(new_name=7)

        self.assertEqual(result, 7)
        self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_both_passed_equal_no_conflict_warning(self):
        fn = _make_wrapped()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(old_name=5, new_name=5)

        self.assertEqual(result, 5)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)

    def test_both_passed_conflict_prefer_old(self):
        fn = _make_wrapped(prefer="old")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(old_name="OLD", new_name="NEW")

        self.assertEqual(result, "OLD")
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertGreaterEqual(len(deprecations), 2)

    def test_both_passed_conflict_prefer_new(self):
        fn = _make_wrapped(prefer="new")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(old_name="OLD", new_name="NEW")

        self.assertEqual(result, "NEW")

    def test_invalid_prefer_raises(self):
        with self.assertRaises(ValueError):
            deprecate_kwargs({"old_name": "new_name"}, prefer="invalid")

    def test_decorator_preserves_function_metadata(self):
        @deprecate_kwargs({"old": "new"})
        def my_fn(new=None, *, old=None):
            """doc."""
            return new

        self.assertEqual(my_fn.__name__, "my_fn")
        self.assertEqual(my_fn.__doc__, "doc.")

    def test_works_on_instance_method(self):
        class Owner:
            @deprecate_kwargs({"old": "new"})
            def method(self, new=None, *, old=None):
                return (self.__class__.__name__, new)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Owner().method(old=123)

        self.assertEqual(result, ("Owner", 123))
