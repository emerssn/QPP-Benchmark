from tira.rest_api_client import Client
from tira.third_party_integrations import ensure_pyterrier_is_loaded

ensure_pyterrier_is_loaded()
import pyterrier as pt
tira = Client()