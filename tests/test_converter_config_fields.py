"""Converter config fields are parameters of their converter (where passed unchanged)."""

import annotationlib
import importlib
import inspect
import re

import pytest

from docler_config import converter_configs


def _config_classes():
    source = inspect.getsource(converter_configs)
    for match in re.finditer(r"^class (\w+Config)\(BaseConverterConfig\):", source, re.M):
        cls = getattr(converter_configs, match.group(1))
        body = inspect.getsource(cls.get_provider)
        imp = re.search(r"from (docler\.converters\.\w+) import (\w+)", body)
        # Only configs whose fields go to the converter unchanged (not e.g. from_config)
        if imp and "get_config_fields()" in body:
            yield pytest.param(cls, imp.group(1), imp.group(2), id=cls.__name__)


@pytest.mark.parametrize(("config_cls", "module", "name"), list(_config_classes()))
def test_config_fields_match_converter(config_cls, module, name):
    try:
        converter = getattr(importlib.import_module(module), name)
    except ImportError as e:
        pytest.skip(f"{module} needs an optional dependency: {e}")
    params = inspect.signature(
        converter.__init__, annotation_format=annotationlib.Format.FORWARDREF
    ).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    fields = set(config_cls.model_fields) - {"type"}
    missing = fields - set(params)
    assert not missing, f"{name}.__init__ lacks config fields {sorted(missing)}"
