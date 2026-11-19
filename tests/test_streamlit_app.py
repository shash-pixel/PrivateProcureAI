import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason=(
        "Streamlit AppTest currently stalls while creating its Windows event loop on Python 3.13."
    ),
)
def test_streamlit_app_renders_the_planning_form() -> None:
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path).run(timeout=15)

    assert not app.exception
    assert app.text_area(key="goal_input")
    assert app.checkbox(key="include_budget")
    assert app.text_area(key="must_have")
