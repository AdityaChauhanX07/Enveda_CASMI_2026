"""Kaggle CLI 2.x routed through www.kaggle.com.

Some sandboxed environments allow www.kaggle.com but not api.kaggle.com. Kaggle serves the same RPC
methods under https://www.kaggle.com/api/v1/<service>/<method>, so this wrapper points the SDK there.

Usage:  python tools/kaggle_www.py <normal kaggle CLI arguments>
        e.g. python tools/kaggle_www.py kernels status adityachauhanx/casmi26-fork-ozertuu-v21
"""
import sys

import kagglesdk.kaggle_env as ke
import kagglesdk.kaggle_http_client as kh

ke._env_to_endpoint[ke.KaggleEnv.PROD] = "https://www.kaggle.com"
kh.KaggleHttpClient._get_request_url = lambda self, service, request: f"{self._endpoint}/api/v1/{service}/{request}"

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
