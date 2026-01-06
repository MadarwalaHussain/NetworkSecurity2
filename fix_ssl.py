"""
SSL Certificate Verification Fix Module - Simplified Version
"""

import sys
import ssl
import os
import warnings
import urllib3
import requests


_ssl_fix_applied = False


def apply_ssl_fix():
    """Apply simplified SSL fix that avoids version conflicts."""
    global _ssl_fix_applied

    if _ssl_fix_applied:
        print("SSL fix already applied, skipping...")
        return

    print("Applying SSL certificate verification fix...")

    # 1. Create unverified SSL context globally
    ssl._create_default_https_context = ssl._create_unverified_context

    # 2. Disable SSL warnings
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    urllib3.disable_warnings()

    # 3. Set environment variables
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

    # 4. Patch requests.Session - simple approach
    _original_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request

    # 5. Patch httpx if available
    try:
        import httpx

        _original_httpx_request = httpx.request

        def _patched_httpx_request(*args, **kwargs):
            kwargs['verify'] = False
            return _original_httpx_request(*args, **kwargs)
        httpx.request = _patched_httpx_request

        _original_httpx_client_init = httpx.Client.__init__

        def _patched_httpx_client_init(self, *args, **kwargs):
            kwargs['verify'] = False
            _original_httpx_client_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_httpx_client_init

        print("✓ httpx patched")
    except ImportError:
        pass

    _ssl_fix_applied = True
    print("✓ SSL fix applied successfully")


def is_ssl_fix_applied():
    """Check if SSL fix has been applied."""
    return _ssl_fix_applied


if os.getenv('AUTO_APPLY_SSL_FIX', '').lower() in ('true', '1', 'yes'):
    apply_ssl_fix()
