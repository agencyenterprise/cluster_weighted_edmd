"""
Patch pykoopman's Polynomial observable for sklearn >= 1.0 compatibility.

sklearn renamed `n_input_features_` to `n_features_in_` but pykoopman's
Polynomial.transform() still checks for the old name via check_is_fitted.

This script patches the installed pykoopman in-place by adding one line
to Polynomial.fit() that aliases the new name back to the old one.

Usage:
    python patches/fix_pykoopman_sklearn.py

Run this once after `pip install -r requirements.txt`.
"""

import os
import pykoopman

poly_path = os.path.join(
    os.path.dirname(pykoopman.__file__),
    "observables", "_polynomial.py"
)

with open(poly_path, "r") as f:
    content = f.read()

PATCH_MARKER = "# PATCH: sklearn compat"
PATCH_LINE = (
    "        self.n_input_features_ = self.n_features_in_  "
    f"{PATCH_MARKER}\n"
)

if PATCH_MARKER in content:
    print(f"Already patched: {poly_path}")
else:
    # Insert after: y_poly_out = super(Polynomial, self).fit(x.real, y)
    target = "y_poly_out = super(Polynomial, self).fit(x.real, y)\n"
    if target not in content:
        print(f"ERROR: Could not find target line in {poly_path}")
        print("  Expected: y_poly_out = super(Polynomial, self).fit(x.real, y)")
        exit(1)

    content = content.replace(target, target + PATCH_LINE)
    with open(poly_path, "w") as f:
        f.write(content)
    print(f"Patched: {poly_path}")
    print(f"  Added: self.n_input_features_ = self.n_features_in_")

# Verify by running a subprocess (ensures fresh import of patched module)
import subprocess, sys
print("\nVerifying...")
verify_code = (
    "from pykoopman.observables import Polynomial; import numpy as np; "
    "obs = Polynomial(degree=2, include_bias=True); "
    "obs.fit(np.random.randn(10, 3)); "
    "r = obs.transform(np.random.randn(5, 3)); "
    "print(f'  Polynomial.fit() + transform(): OK (output shape {r.shape})')"
)
result = subprocess.run([sys.executable, "-c", verify_code], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout.strip())
else:
    print("  VERIFICATION FAILED:")
    print(result.stderr)
    sys.exit(1)
