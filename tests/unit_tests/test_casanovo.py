import datetime
import functools
import hashlib
import io
import logging
import os
import pathlib
import re
import shutil
import tempfile
import unittest.mock
import warnings
from typing import NoReturn

import click.testing
import github
import pytest
import requests

from casanovo import casanovo, utils
from casanovo.casanovo import _SharedFileIOParams
from casanovo.config import Config


def test_version():
    """Check that the version is not None."""
    assert casanovo.__version__ is not None


@pytest.mark.parametrize(
    ("model_file", "expectation", "log_message"),
    [
        (None, None, "must also be provided"),
        (
            "https://github.com/Noble-Lab/casanovo",
            pytest.raises(ValueError, match="cannot be loaded from a URL"),
            None,
        ),
        (
            "nonexistent_file.ckpt",
            pytest.raises(ValueError, match="must point to an existing file"),
            None,
        ),
    ],
)
def test_is_valid_model(model_file, expectation, log_message, caplog):
    if expectation is None:
        casanovo._is_valid_model(model_file, continue_training=True)
        assert log_message in caplog.text
    else:
        with expectation:
            casanovo._is_valid_model(model_file, continue_training=True)


class MockAsset:
    def __init__(self, file_name):
        self.name = file_name
        self.browser_download_url = f"http://example.com/{file_name}"


class MockGithub:
    def __init__(self, releases):
        self.releases = releases

    def get_repo(self, repo_name):
        return MockRepo()


class MockRelease:
    def __init__(self, tag_name, assets):
        self.tag_name = tag_name
        self.assets = [MockAsset(asset) for asset in assets]

    def get_assets(self):
        return self.assets


class MockRepo:
    def __init__(
        self,
        release_dict={
            "v3.0.0": [
                "casanovo_orbitrap_v3-0-0.ckpt",
                "casanovo_timstof_v3-0-0.ckpt",
                "casanovo_non-enzy.ckpt",
                "v3.0.0.zip",
                "v3.0.0.tar.gz",
            ],
            "v3.1.0": ["v3.1.0.zip", "v3.1.0.tar.gz"],
            "v3.2.0": ["v3.2.0.zip", "v3.2.0.tar.gz"],
            "v3.3.0": ["v3.3.0.zip", "v3.3.0.tar.gz"],
            "v4.0.0": [
                "casanovo_orbitrap_v4-0-0.ckpt",
                "casanovo_timstof_v4-0-0.ckpt",
                "casanovo_orbitrap-tmt_v4-0-0.ckpt",
                "v4.0.0.zip",
                "v4.0.0.tar.gz",
            ],
        },
    ):
        self.releases = [
            MockRelease(tag_name, assets)
            for tag_name, assets in release_dict.items()
        ]

    def get_releases(self):
        return self.releases


class MockResponseGet:
    file_content = b"fake model weights content"

    class MockRaw(io.BytesIO):
        def read(self, *args, **kwargs):
            return super().read(*args)

    class FailingRaw(io.BytesIO):
        def read(self, *args, **kwargs):
            if self.tell() == 0:
                return super().read(4)
            raise requests.Timeout

    def __init__(self, fail_after_first_chunk=False):
        self.request_counter = 0
        self.is_ok = True
        self.timeouts = []
        self.fail_after_first_chunk = fail_after_first_chunk

    def raise_for_status(self):
        if not self.is_ok:
            raise requests.HTTPError

    def __call__(self, url, stream=True, allow_redirects=True, timeout=None):
        self.request_counter += 1
        self.timeouts.append(timeout)
        response = unittest.mock.MagicMock()
        response.raise_for_status = self.raise_for_status
        response.headers = {"Content-Length": str(len(self.file_content))}
        if self.fail_after_first_chunk:
            response.raw = MockResponseGet.FailingRaw(self.file_content)
        else:
            response.raw = MockResponseGet.MockRaw(self.file_content)
        return response


class MockResponseHead:
    def __init__(self):
        self.last_modified = None
        self.is_ok = True
        self.fail = False
        self.timeouts = []

    def __call__(self, url, timeout=None):
        self.timeouts.append(timeout)
        if self.fail:
            raise requests.ConnectionError

        response = unittest.mock.MagicMock()
        response.headers = dict()
        response.ok = self.is_ok
        if self.last_modified is not None:
            response.headers["Last-Modified"] = self.last_modified

        return response


@pytest.mark.parametrize(
    ("model_arg", "expected_filename"),
    [
        (None, "casanovo_orbitrap_v3-0-0.ckpt"),
        ("timstof", "casanovo_timstof_v3-0-0.ckpt"),
    ],
)
def test_setup_model(monkeypatch, model_arg, expected_filename):
    test_releases = ["3.0.0", "3.0.999", "3.999.999"]
    mock_get = MockResponseGet()
    mock_github = functools.partial(MockGithub, test_releases)
    version = "3.0.0"

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)
        filename = pathlib.Path(tmp_dir) / expected_filename

        assert not filename.is_file()
        _, result_path = casanovo.setup_model(
            model_arg, None, None, None, False
        )
        assert result_path.resolve() == filename.resolve()
        assert filename.is_file()
        assert mock_get.request_counter == 1
        os.remove(result_path)

        assert not filename.is_file()
        _, result = casanovo.setup_model(None, None, None, None, True)
        assert result is None
        assert not filename.is_file()
        assert mock_get.request_counter == 1

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        cache_file_name = "model_weights.ckpt"
        file_url = f"http://www.example.com/{cache_file_name}"
        url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
        cache_dir = pathlib.Path(tmp_dir)
        cache_file_dir = cache_dir / url_hash
        cache_file_path = cache_file_dir / cache_file_name

        assert not cache_file_path.is_file()
        _, result_path = casanovo.setup_model(
            file_url, None, None, None, False
        )
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 2
        os.remove(result_path)

        assert not cache_file_path.is_file()
        _, result_path = casanovo.setup_model(
            file_url, None, None, None, False
        )
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

    # Test model is file
    with (
        monkeypatch.context() as mnk,
        tempfile.NamedTemporaryFile(suffix=".ckpt") as temp_file,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        temp_file_path = temp_file.name
        _, result = casanovo.setup_model(
            temp_file_path, None, None, None, False
        )
        assert mock_get.request_counter == 3
        assert result == pathlib.Path(temp_file_path)

        _, result = casanovo.setup_model(
            temp_file_path, None, None, None, True
        )
        assert mock_get.request_counter == 3
        assert result == pathlib.Path(temp_file_path)

    # Test model is neither a URL or File
    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        with pytest.raises(ValueError):
            casanovo.setup_model("FooBar", None, None, None, False)

        assert mock_get.request_counter == 3

        with pytest.raises(ValueError):
            casanovo.setup_model("FooBar", None, None, None, False)

        assert mock_get.request_counter == 3


@pytest.mark.parametrize("model_arg", [None, "timstof"])
def test_rate_limit_exception(monkeypatch, model_arg):
    def mock_get_model_weights(selector, cache_dir, casanovo_version):
        raise github.RateLimitExceededException(403, "rate limit exceeded", {})

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr(casanovo, "__version__", "3.0.0")
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(casanovo, "_get_model_weights", mock_get_model_weights)

        with pytest.raises(
            PermissionError, match="GitHub API rate limit exceeded"
        ):
            casanovo.setup_model(model_arg, None, None, None, False)


@pytest.mark.parametrize(
    "selector, expected_filename",
    [
        ("orbitrap", "casanovo_orbitrap_v3-0-0.ckpt"),
        ("timstof", "casanovo_timstof_v3-0-0.ckpt"),
    ],
)
def test_get_model_weights(monkeypatch, selector, expected_filename):
    test_releases = ["3.0.0", "3.0.999", "3.999.999"]
    mock_get = MockResponseGet()
    mock_github = functools.partial(MockGithub, test_releases)

    # Matching versions: exact, same minor, same major.
    for version in test_releases:
        with (
            monkeypatch.context() as mnk,
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            version_tup = tuple(
                int(x) if x else 0 for x in utils.split_version(version)
            )
            mnk.setattr(casanovo, "__version__", version)
            mnk.setattr(
                "appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir
            )
            mnk.setattr(github, "Github", mock_github)
            mnk.setattr(requests, "get", mock_get)

            tmp_path = pathlib.Path(tmp_dir)
            filename = tmp_path / expected_filename
            assert not filename.is_file()
            result_path = casanovo._get_model_weights(
                selector, tmp_path, version_tup
            )
            assert result_path == filename
            assert filename.is_file()
            # Second call should use cache, no extra download.
            result_path = casanovo._get_model_weights(
                selector, tmp_path, version_tup
            )
            assert result_path == filename

    # Impossible to find model weights for (i) full version mismatch and (ii)
    # major version mismatch.
    for version in ["999.999.999", "999.0.0"]:
        with (
            monkeypatch.context() as mnk,
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            version_tup = tuple(
                int(x) if x else 0 for x in utils.split_version(version)
            )

            mnk.setattr(casanovo, "__version__", version)
            mnk.setattr(github, "Github", mock_github)
            mnk.setattr(requests, "get", mock_get)
            with pytest.raises(ValueError):
                casanovo._get_model_weights(
                    selector,
                    pathlib.Path(tmp_dir),
                    version_tup,
                )

    # Test GitHub API rate limit.
    def request(self, *args, **kwargs):
        raise github.RateLimitExceededException(
            403, "API rate limit exceeded", None
        )

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr("github.Requester.Requester.requestJsonAndCheck", request)
        mnk.setattr(requests, "get", mock_get)
        mock_get.request_counter = 0
        with pytest.raises(github.RateLimitExceededException):
            casanovo._get_model_weights(
                selector, pathlib.Path(tmp_dir), (3, 0, 0)
            )

        assert mock_get.request_counter == 0


@pytest.mark.parametrize(
    "selector, candidates, expected",
    [
        # Exact match
        ("orbitrap", ["orbitrap", "timstof"], "orbitrap"),
        ("timstof", ["orbitrap", "timstof"], "timstof"),
        # Case-insensitive exact
        ("Orbitrap", ["orbitrap", "timstof"], "orbitrap"),
        ("TIMSTOF", ["orbitrap", "timstof"], "timstof"),
        # Prefix match
        ("tims", ["orbitrap", "timstof"], "timstof"),
        ("orbit", ["orbitrap"], "orbitrap"),
        # Substring match
        ("tmt", ["orbitrap", "orbitrap-tmt"], "orbitrap-tmt"),
        # Hyphen/separator stripped
        ("orbitrap-tmt", ["orbitrap", "orbitrap-tmt"], "orbitrap-tmt"),
    ],
)
def test_resolve_selector_success(selector, candidates, expected):
    assert casanovo._resolve_selector(selector, candidates) == expected


@pytest.mark.parametrize(
    "selector, candidates",
    [
        # Ambiguous prefix
        ("orbit", ["orbitrap", "orbitrap-tmt"]),
        # Ambiguous substring
        ("rap", ["orbitrap", "orbitrap-tmt"]),
        # Unknown
        ("xyz", ["orbitrap", "timstof"]),
        ("foobar", ["orbitrap", "timstof"]),
    ],
)
def test_resolve_selector_failure(selector, candidates):
    with pytest.raises(ValueError):
        casanovo._resolve_selector(selector, candidates)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("casanovo_orbitrap_v5-0-0.ckpt", ("orbitrap", (5, 0, 0))),
        ("casanovo_timstof_v5-2-0.ckpt", ("timstof", (5, 2, 0))),
        ("casanovo_orbitrap-tmt_v6-0-1.ckpt", ("orbitrap-tmt", (6, 0, 1))),
        # Path prefix is stripped correctly
        ("/some/dir/casanovo_orbitrap_v3-0-0.ckpt", ("orbitrap", (3, 0, 0))),
        ("casanovo_Orbitrap_v3-0-0.ckpt", ("orbitrap", (3, 0, 0))),
    ],
)
def test_parse_ckpt_valid(filename, expected):
    assert casanovo._parse_ckpt(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "casanovo_massivekb.ckpt",  # old format, no version
        "casanovo_timstof.ckpt",  # old format, no version
        "casanovo_non-enzy.checkpt",  # wrong extension
        "casanovo__v3-0-0.ckpt",  # empty model ID
        "casanovo_orbitrap_v3.0.0.ckpt",  # dots instead of dashes in version
        "v3.0.0.zip",
        "model.ckpt",
    ],
)
def test_parse_ckpt_invalid(filename):
    assert casanovo._parse_ckpt(filename) is None


def test_best_ckpt_no_cross_family():
    """Version resolution must never silently fall back across model families."""
    # Caller is responsible for pre-filtering by family; _best_ckpt only
    # sees one family's checkpoints. Confirm it returns None rather than
    # picking something from a different family if the list is empty.
    assert casanovo._best_ckpt([], (4, 3, 2)) is None


def test_get_weights_from_url(monkeypatch):
    file_url = "http://example.com/model_weights.ckpt"

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        mock_get = MockResponseGet()
        mock_head = MockResponseHead()
        mnk.setattr(requests, "get", mock_get)
        mnk.setattr(requests, "head", mock_head)

        cache_dir = pathlib.Path(tmp_dir)
        url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
        cache_file_name = "model_weights.ckpt"
        cache_file_dir = cache_dir / url_hash
        cache_file_path = cache_file_dir / cache_file_name

        # Test downloading and caching the file
        assert not cache_file_path.is_file()
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 1
        assert mock_get.timeouts == [casanovo._MODEL_WEIGHT_REQUEST_TIMEOUT]

        # Test that cached file is used
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 1
        assert mock_head.timeouts == [casanovo._MODEL_WEIGHT_REQUEST_TIMEOUT]

        # Test force downloading the file
        result_path = casanovo._get_weights_from_url(
            file_url, cache_dir, force_download=True
        )
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 2
        assert mock_get.timeouts[-1] == casanovo._MODEL_WEIGHT_REQUEST_TIMEOUT

        # Test that file is re-downloaded if last modified is newer than
        # file last modified
        # NOTE: Assuming test takes < 1 year to run
        curr_utc = datetime.datetime.now().astimezone(datetime.timezone.utc)
        mock_head.last_modified = (
            curr_utc + datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

        # Test file is not redownloaded if its newer than upstream file
        mock_head.last_modified = (
            curr_utc - datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

        # Test that error is raised if file get response is not OK
        mock_get.is_ok = False
        with pytest.raises(requests.HTTPError):
            casanovo._get_weights_from_url(
                file_url, cache_dir, force_download=True
            )
        mock_get.is_ok = True
        assert mock_get.request_counter == 4

        # Test that cached file is used if head requests yields non-ok status
        # code, even if upstream file is newer
        mock_head.is_ok = False
        mock_head.last_modified = (
            curr_utc + datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 4
        mock_head.is_ok = True

        # Test that cached file is used if head request fails
        mock_head.fail = True
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 4
        mock_head.fail = False

        # Test invalid URL
        with pytest.raises(ValueError):
            bad_url = "foobar"
            casanovo._get_weights_from_url(bad_url, cache_dir)


def test_get_weights_from_url_discards_partial_download(monkeypatch):
    file_url = "http://example.com/model_weights.ckpt"

    with (
        monkeypatch.context() as mnk,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        failing_get = MockResponseGet(fail_after_first_chunk=True)
        mnk.setattr(requests, "get", failing_get)

        cache_dir = pathlib.Path(tmp_dir)
        url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
        cache_file_dir = cache_dir / url_hash
        cache_file_path = cache_file_dir / "model_weights.ckpt"

        with pytest.raises(requests.Timeout):
            casanovo._get_weights_from_url(file_url, cache_dir)

        assert not cache_file_path.exists()
        assert not list(cache_file_dir.glob("model_weights.ckpt.*.tmp"))

        successful_get = MockResponseGet()
        mock_head = MockResponseHead()
        mnk.setattr(requests, "get", successful_get)
        mnk.setattr(requests, "head", mock_head)

        result_path = casanovo._get_weights_from_url(file_url, cache_dir)

        assert result_path.resolve() == cache_file_path.resolve()
        assert cache_file_path.read_bytes() == MockResponseGet.file_content
        assert successful_get.request_counter == 1


def test_download_weights_discards_temp_file_on_keyboard_interrupt(
    monkeypatch, tmp_path
):
    download_path = tmp_path / "model_weights.ckpt"
    mock_get = MockResponseGet()
    monkeypatch.setattr(requests, "get", mock_get)

    def fail_copyfileobj(_source, destination) -> NoReturn:
        destination.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copyfileobj", fail_copyfileobj)

    with pytest.raises(KeyboardInterrupt):
        casanovo._download_weights(
            "http://example.com/model_weights.ckpt", download_path
        )

    assert not download_path.exists()
    assert not list(tmp_path.glob("model_weights.ckpt.*.tmp"))


def test_is_valid_url():
    assert casanovo._is_valid_url("https://www.washington.edu/")
    assert not casanovo._is_valid_url("foobar")


def test_setup_output(tmp_path, monkeypatch):
    with monkeypatch.context() as mnk:
        mnk.setattr(pathlib.Path, "cwd", lambda: tmp_path)
        output_path, output_root = casanovo._setup_output(
            None, None, False, "info"
        )
        assert output_path.resolve() == tmp_path.resolve()
        assert re.fullmatch(r"casanovo_\d+", output_root) is not None

        target_path = tmp_path / "foo"
        output_path, output_root = casanovo._setup_output(
            str(target_path), "bar", False, "info"
        )

        assert output_path.resolve() == target_path.resolve()
        assert output_root == "bar"


def test_setup_logging_warnings(tmp_path):
    """Warnings are logged once, on a single line."""
    log_path = tmp_path / "test.log"
    casanovo.setup_logging(log_path, "info")
    warnings.warn("Hello\n World")
    lines = log_path.read_text().splitlines()
    matching = []
    for line in lines:
        if "Hello" in line:
            matching.append(line)
    assert len(matching) == 1
    assert "Hello World" in matching[0]


def test_setup_logging_repeated_calls(tmp_path):
    """Repeated calls replace the previous handlers instead of adding more."""
    root_logger = logging.getLogger()
    n_handlers = len(root_logger.handlers)
    for i in range(3):
        casanovo.setup_logging(tmp_path / f"run{i}.log", "info")
    assert len(root_logger.handlers) == n_handlers + 2

    logging.getLogger("casanovo").info("hello")
    assert "hello" not in (tmp_path / "run0.log").read_text()
    assert (tmp_path / "run2.log").read_text().count("hello") == 1


def test_setup_logging_reinstalls_warning_capture(tmp_path):
    """Warnings are logged even if the warnings hook was replaced since an
    earlier call, as pytest does between tests."""
    casanovo.setup_logging(tmp_path / "first.log", "info")
    log_path = tmp_path / "second.log"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        casanovo.setup_logging(log_path, "info")
        warnings.warn("hello")
    assert "hello" in log_path.read_text()


@pytest.mark.parametrize(
    "args,should_fail",
    [
        # --- output_dir ---
        (lambda tmp_path: ["--output_dir", str(tmp_path)], False),
        # --- output_root ---
        (lambda tmp_path: ["--output_root", "my_root"], False),
        (
            lambda tmp_path: ["--output_root", str(tmp_path)],
            False,
        ),  # output_root as a dir path (string)
        # --- force_overwrite ---
        (lambda tmp_path: ["--force_overwrite"], False),
        # --- verbosity ---
        (lambda tmp_path: ["--verbosity", "debug"], False),
        (lambda tmp_path: ["--verbosity", "INFO"], False),  # case-insensitive
        (lambda tmp_path: ["--verbosity", "nope"], True),
    ],
)
def test_shared_file_io_params_validation(tmp_path, args, should_fail):
    @click.command(cls=_SharedFileIOParams)
    def dummy(*args, **kwargs):
        click.echo("ok")

    argv = args(tmp_path)
    res = click.testing.CliRunner().invoke(dummy, argv)

    if should_fail:
        assert res.exit_code != 0
        assert "Error" in res.stderr
        assert "ok" not in res.stdout
    else:
        assert res.exit_code == 0
        assert "Error" not in res.stderr
        assert "ok" in res.stdout


def test_train_cli_tracking_peak_path(tmp_path, mgf_small, monkeypatch):
    """-t/--tracking_peak_path CLI option is forwarded to ModelRunner.train()."""
    from casanovo.denovo.model_runner import ModelRunner as _ModelRunner

    captured = {}

    def fake_train(self, train_pp, valid_pp, ckpt=None, tracking_pp=()):
        captured["tracking"] = tracking_pp

    monkeypatch.setattr(_ModelRunner, "train", fake_train)
    monkeypatch.setattr(casanovo, "_is_valid_model", lambda *a, **kw: None)
    monkeypatch.setattr(
        casanovo, "_setup_output", lambda *a, **kw: (tmp_path, "out")
    )
    monkeypatch.setattr(
        casanovo, "setup_model", lambda *a, **kw: (Config(), None)
    )
    monkeypatch.setattr(utils, "log_system_info", lambda: None)
    monkeypatch.setattr(utils, "log_run_report", lambda **kw: None)

    result = click.testing.CliRunner().invoke(
        casanovo.main,
        ["train", "-t", str(mgf_small), str(mgf_small)],
    )
    assert result.exit_code == 0, result.output
    assert str(mgf_small) in captured.get("tracking", ())
