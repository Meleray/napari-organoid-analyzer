from napari_organoid_analyzer import _utils as utils
from napari_organoid_analyzer import settings
import json

SESSION_VARS = {}

def set_session_var(name, value):
    SESSION_VARS.update({name: value})
    utils.write_to_json(settings.SETTINGS_DIR / 'session_vars.json', SESSION_VARS)

def load_cached_settings():
    if (settings.SETTINGS_DIR / 'session_vars.json').exists():
        with open(settings.SETTINGS_DIR / 'session_vars.json', 'r') as f:
            SESSION_VARS.update(json.load(f))
