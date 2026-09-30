"""The suite configs shown in the docs are real: every ```protobuf block
titled as a .pbtxt parses as a ProfilerSuiteConfig (fields that were
removed, e.g. sampling_interval_ns or pids, fail to parse)."""

import os
import re

import pytest
from google.protobuf import text_format

import profiler_config_pb2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = ["README.md"] + [os.path.join("docs", f) for f in sorted(os.listdir(os.path.join(REPO, "docs")))
                        if f.endswith(".md")]
BLOCK = re.compile(r'^```protobuf title:"([^"]*\.pbtxt[^"]*)"\n(.*?)^```', re.M | re.S)


def _blocks():
    for doc in DOCS:
        for m in BLOCK.finditer(open(os.path.join(REPO, doc)).read()):
            yield pytest.param(m.group(2), id=f"{doc}:{m.group(1)}")


BLOCKS = list(_blocks())


def test_docs_show_suite_configs():
    assert len(BLOCKS) >= 2, [b.id for b in BLOCKS]


@pytest.mark.parametrize("text", BLOCKS)
def test_doc_config_parses(text):
    text_format.Parse(text, profiler_config_pb2.ProfilerSuiteConfig())
