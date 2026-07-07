# Copyright (c) Open-CD. All rights reserved.
import warnings
import mmcv
import mmengine
import mmseg

try:
    import mmdet
except ImportError:
    mmdet = None

from .version import __version__, version_info

__all__ = ['__version__', 'version_info']
