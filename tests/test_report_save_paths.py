"""Tests for report save-path resolution and the fsspec-backed report writer.

Regression coverage for #807 / PR #858: cloud URIs (e.g. ``s3://bucket/reports``)
must not be wrapped in :class:`pathlib.Path`, which would collapse the scheme's
``://`` into ``:/`` and silently reroute the write to a local directory.
"""

from pathlib import Path

import pytest

# fsspec is a declared dependency; the writer imports it lazily. Skip cleanly if
# the environment somehow lacks it rather than erroring at collection time.
fsspec = pytest.importorskip("fsspec")

from cli.main import resolve_save_target, save_report_to_disk  # noqa: E402


@pytest.mark.unit
class TestResolveSaveTarget:
    def test_local_path_becomes_path(self):
        result = resolve_save_target("/home/user/reports/AAPL")
        assert isinstance(result, Path)
        assert str(result) == "/home/user/reports/AAPL"

    def test_relative_local_path_becomes_path(self):
        result = resolve_save_target("reports/AAPL")
        assert isinstance(result, Path)

    @pytest.mark.parametrize(
        "uri",
        [
            "s3://my-bucket/reports",
            "gcs://my-bucket/reports",
            "gs://my-bucket/reports",
            "az://my-container/reports",
            "memory://reports",
        ],
    )
    def test_cloud_uri_stays_a_string_with_double_slash(self, uri):
        result = resolve_save_target(uri)
        assert isinstance(result, str)
        # The scheme separator must survive untouched.
        assert "://" in result
        assert result == uri

    def test_cloud_uri_not_collapsed_by_path(self):
        # The exact bug being guarded: Path() would turn this into "s3:/bucket/x".
        uri = "s3://bucket/x"
        assert resolve_save_target(uri) == uri
        assert str(Path(uri)) != uri  # sanity: Path really does collapse it


def _minimal_state():
    return {
        "market_report": "market body",
        "sentiment_report": "sentiment body",
    }


@pytest.mark.unit
class TestSaveReportToDisk:
    def test_writes_tree_for_local_path(self, tmp_path):
        target = tmp_path / "AAPL_run"
        complete = save_report_to_disk(_minimal_state(), "AAPL", target)

        assert (target / "1_analysts" / "market.md").read_text() == "market body"
        assert (target / "1_analysts" / "sentiment.md").read_text() == "sentiment body"
        assert complete == f"{str(target).rstrip('/')}/complete_report.md"
        assert Path(complete).exists()

    def test_sentiment_label_is_consistent(self, tmp_path):
        target = tmp_path / "AAPL_run"
        complete = save_report_to_disk(_minimal_state(), "AAPL", target)
        body = Path(complete).read_text()
        # Finding #3: consolidated report must label the social analyst the same
        # way the on-disk file (sentiment.md) and display_complete_report do.
        assert "### Sentiment Analyst" in body
        assert "### Social Analyst" not in body

    def test_memory_uri_string_routes_to_fsspec_not_local(self):
        # A non-local fsspec URI passed as a string must reach the in-memory FS,
        # proving the write is NOT silently rerouted to the local cwd.
        mem = fsspec.filesystem("memory")
        base = "memory://reports/AAPL_run"
        # Clean any leftover from a prior run.
        if mem.exists("/reports"):
            mem.rm("/reports", recursive=True)

        complete = save_report_to_disk(_minimal_state(), "AAPL", base)

        assert complete == "memory://reports/AAPL_run/complete_report.md"
        assert mem.exists("/reports/AAPL_run/1_analysts/market.md")
        assert mem.exists("/reports/AAPL_run/complete_report.md")
        with mem.open("/reports/AAPL_run/1_analysts/market.md", "r") as f:
            assert f.read() == "market body"
        # And it must NOT have leaked into a local "memory:" directory.
        assert not Path("memory:").exists()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
