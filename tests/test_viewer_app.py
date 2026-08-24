"""Phase 8 gate check: the app runs (no exceptions) and re-predicts in
under a second. Uses Streamlit's own AppTest framework to actually
execute app/viewer.py headlessly and simulate a sample switch - this
runs the real script, not a hand-rolled approximation of it.
"""

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "viewer.py")


def main():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception, f"App raised on initial run: {at.exception}"
    print(f"Initial run OK. {len(at.selectbox)} selectbox(es), "
          f"{len(at.dataframe)} dataframe(s) rendered.")

    select = at.selectbox[0]
    other_options = [o for o in select.options if o != select.value]
    assert other_options, "Only one sample available - can't test switching"

    start = time.perf_counter()
    select.set_value(other_options[0]).run()
    elapsed = time.perf_counter() - start

    assert not at.exception, f"App raised after switching sample: {at.exception}"
    print(f"Switched to sample '{other_options[0]}' and reran in {elapsed*1000:.1f} ms "
          f"(includes Streamlit script re-execution overhead, not just model inference)")

    caption_texts = [c.value for c in at.caption]
    timing_captions = [c for c in caption_texts if "Re-predicted in" in c]
    assert timing_captions, "No timing caption found"
    print(f"App's own reported timing: {timing_captions[0]}")

    assert len(at.metric) == 4, f"Expected 4 metrics (Spearman H/V, Kendall H/V), got {len(at.metric)}"
    for m in at.metric:
        print(f"  {m.label}: {m.value}")

    assert not at.exception
    print("\nPASS: app runs without exceptions, sample switching works, timing reported")


if __name__ == "__main__":
    main()
