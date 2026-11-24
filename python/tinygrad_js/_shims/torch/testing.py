import numpy as np
def assert_close(actual, expected, rtol=1.3e-6, atol=1e-5, equal_nan=False, check_dtype=True, msg=None, **k):
  a = actual.detach().numpy() if hasattr(actual, "detach") else np.asarray(actual)
  b = expected.detach().numpy() if hasattr(expected, "detach") else np.asarray(expected)
  np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=msg or "")
def assert_allclose(actual, expected, rtol=1e-7, atol=0, **k): assert_close(actual, expected, rtol=rtol, atol=atol)
