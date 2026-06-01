"""E2B execution → Letta multimodal tool return formatting."""

from types import SimpleNamespace

from letta.schemas.letta_message_content import ImageContent, TextContent
from letta.services.tool_executor.e2b_result_format import e2b_execution_to_func_return


def _execution(*, results=None, stdout=None, stderr=None, error=None):
    return SimpleNamespace(
        results=results or [],
        logs=SimpleNamespace(stdout=stdout or [], stderr=stderr or []),
        error=error,
    )


def test_png_result_returns_image_block():
    result = SimpleNamespace(text=None, png="aGVsbG8=", jpeg=None)
    out = e2b_execution_to_func_return(_execution(results=[result]))
    assert isinstance(out, list)
    assert any(isinstance(p, ImageContent) for p in out)
    img = next(p for p in out if isinstance(p, ImageContent))
    assert img.source.media_type == "image/png"
    assert img.source.data == "aGVsbG8="


def test_text_only_returns_dict():
    result = SimpleNamespace(text="42", png=None, jpeg=None)
    out = e2b_execution_to_func_return(_execution(results=[result]))
    assert isinstance(out, dict)
    assert out["results"] == ["42"]


def test_matplotlib_style_png_with_logs():
    result = SimpleNamespace(text=None, png="plotb64", jpeg=None)
    out = e2b_execution_to_func_return(
        _execution(results=[result], stdout=["pre-plot\n"], stderr=[])
    )
    assert isinstance(out, list)
    assert isinstance(out[0], TextContent)
    assert "pre-plot" in out[0].text
    assert isinstance(out[-1], ImageContent)


def test_execution_error_included_in_text():
    err = SimpleNamespace(name="ValueError", value="bad", traceback="tb")
    out = e2b_execution_to_func_return(_execution(results=[], error=err))
    assert isinstance(out, dict)
    assert out["error"]["name"] == "ValueError"


def test_duplicate_png_across_e2b_results_deduped():
    """Display + main result often carry the same PNG once per run_code cell."""
    display = SimpleNamespace(text=None, png="sameplotb64", jpeg=None, is_main_result=False)
    main = SimpleNamespace(text=None, png="sameplotb64", jpeg=None, is_main_result=True)
    out = e2b_execution_to_func_return(_execution(results=[display, main]))
    assert isinstance(out, list)
    images = [p for p in out if isinstance(p, ImageContent)]
    assert len(images) == 1


def test_two_distinct_plots_kept():
    a = SimpleNamespace(text=None, png="plotA", jpeg=None)
    b = SimpleNamespace(text=None, png="plotB", jpeg=None)
    out = e2b_execution_to_func_return(_execution(results=[a, b]))
    images = [p for p in out if isinstance(p, ImageContent)]
    assert len(images) == 2


def test_single_result_prefers_png_over_jpeg():
    result = SimpleNamespace(text=None, png="pngbytes", jpeg="jpegbytes")
    out = e2b_execution_to_func_return(_execution(results=[result]))
    images = [p for p in out if isinstance(p, ImageContent)]
    assert len(images) == 1
    assert images[0].source.media_type == "image/png"
