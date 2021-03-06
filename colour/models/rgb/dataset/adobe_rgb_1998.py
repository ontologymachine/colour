#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adobe RGB (1998) Colourspace
============================

Defines the *Adobe RGB (1998)* colourspace:

-   :attr:`ADOBE_RGB_1998_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Adobe Systems. (2005). Adobe RGB (1998) Color Image Encoding.
        Retrieved from http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import RGB_Colourspace, function_gamma

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ADOBE_RGB_1998_PRIMARIES', 'ADOBE_RGB_1998_ILLUMINANT',
    'ADOBE_RGB_1998_WHITEPOINT', 'ADOBE_RGB_1998_TO_XYZ_MATRIX',
    'XYZ_TO_ADOBE_RGB_1998_MATRIX', 'ADOBE_RGB_1998_COLOURSPACE'
]

ADOBE_RGB_1998_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.2100, 0.7100],
     [0.1500, 0.0600]])  # yapf: disable
"""
*Adobe RGB (1998)* colourspace primaries.

ADOBE_RGB_1998_PRIMARIES : ndarray, (3, 2)
"""

ADOBE_RGB_1998_ILLUMINANT = 'D65'
"""
*Adobe RGB (1998)* colourspace whitepoint name as illuminant.

ADOBE_RGB_1998_ILLUMINANT : unicode
"""

ADOBE_RGB_1998_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
    [ADOBE_RGB_1998_ILLUMINANT])  # yapf: disable
"""
*Adobe RGB (1998)* colourspace whitepoint.

ADOBE_RGB_1998_WHITEPOINT : ndarray
"""

ADOBE_RGB_1998_TO_XYZ_MATRIX = np.array(
    [[0.57667, 0.18556, 0.18823],
     [0.29734, 0.62736, 0.07529],
     [0.02703, 0.07069, 0.99134]])  # yapf: disable
"""
*Adobe RGB (1998)* colourspace to *CIE XYZ* tristimulus values matrix defined
as per [1].

ADOBE_RGB_1998_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ADOBE_RGB_1998_MATRIX = np.array(
    [[2.04159, -0.56501, -0.34473],
     [-0.96924, 1.87597, 0.04156],
     [0.01344, -0.11836, 1.01517]])  # yapf: disable
"""
*CIE XYZ* tristimulus values to *Adobe RGB (1998)* colourspace matrix.

XYZ_TO_ADOBE_RGB_1998_MATRIX : array_like, (3, 3)
"""

ADOBE_RGB_1998_COLOURSPACE = RGB_Colourspace(
    'Adobe RGB (1998)',
    ADOBE_RGB_1998_PRIMARIES,
    ADOBE_RGB_1998_WHITEPOINT,
    ADOBE_RGB_1998_ILLUMINANT,
    ADOBE_RGB_1998_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_RGB_1998_MATRIX,
    partial(function_gamma, exponent=1 / (563 / 256)),
    partial(function_gamma, exponent=563 / 256))  # yapf: disable
"""
*Adobe RGB (1998)* colourspace.

ADOBE_RGB_1998_COLOURSPACE : RGB_Colourspace
"""
