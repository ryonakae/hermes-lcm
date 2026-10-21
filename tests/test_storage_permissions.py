"""Regression coverage for private SQLite storage artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hermes_lcm.db_bootstrap as db_bootstrap_module
import hermes_lcm.maintenance as maintenance_module
import hermes_lcm.sqlite_util as sqlite_util_module
from hermes_lcm.maintenance import backup_database, rotate_backup_database
from hermes_lcm.store import MessageStore, build_message_fts_spec


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX mode semantics")


@contextmanager
def _process_umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _sqlite_artifacts(path: Path) -> list[Path]:
    return [path, *(path.with_name(path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES)]


def _assert_private_sqlite_artifacts(path: Path) -> None:
    artifacts = [artifact for artifact in _sqlite_artifacts(path) if artifact.exists()]
    assert artifacts
    assert {artifact.name: _mode(artifact) for artifact in artifacts} == {
        artifact.name: 0o600 for artifact in artifacts
    }


def _seed_searchable_store(db_path: Path) -> None:
    store = MessageStore(db_path)
    try:
        store.append(
            "sidecar-race",
            {"role": "user", "content": "durable sidecar lifecycle token"},
        )
        store.commit()
    finally:
        store.close()


def _assert_searchable_store_integrity(store: MessageStore) -> None:
    conn = store.connection
    assert conn is not None
    assert conn.execute("SELECT content FROM messages").fetchone()[0] == (
        "durable sidecar lifecycle token"
    )
    assert store.search("lifecycle", limit=10)[0]["content"] == (
        "durable sidecar lifecycle token"
    )
    assert db_bootstrap_module.check_external_content_fts_integrity(
        conn, build_message_fts_spec()
    )["status"] == "pass"
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _assert_writer_blocked(db_path: Path) -> None:
    probe = subprocess.run(
        [sys.executable, "-c", """
import sqlite3
import sys
conn = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    conn.execute('BEGIN IMMEDIATE')
except sqlite3.OperationalError as exc:
    if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY:
        raise
    print('blocked')
else:
    conn.rollback()
    print('acquired')
finally:
    conn.close()
""", str(db_path)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert probe.stdout.strip() == "blocked", probe.stdout + probe.stderr


@pytest.mark.parametrize("mode", [0o600, 0o644])
@pytest.mark.parametrize("journal_mode", ["WAL", "DELETE"])
@pytest.mark.parametrize("operation", ["prepare", "restrict"])
def test_permission_changes_preserve_live_sqlite_write_lock(
    tmp_path, mode, journal_mode, operation,
):
    db_path = tmp_path / "lcm.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        assert conn.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0] == journal_mode.lower()
        conn.execute("CREATE TABLE lock_probe(value TEXT)")
        conn.execute("INSERT INTO lock_probe VALUES ('original')")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE lock_probe SET value='uncommitted'")
        for artifact in _sqlite_artifacts(db_path):
            if artifact.exists():
                artifact.chmod(mode)
        _assert_writer_blocked(db_path)

        if operation == "prepare":
            sqlite_util_module._prepare_private_sqlite_file(db_path)
        else:
            sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

        _assert_private_sqlite_artifacts(db_path)
        _assert_writer_blocked(db_path)
        conn.commit()
    finally:
        conn.close()
    with sqlite3.connect(db_path) as fresh:
        assert fresh.execute("SELECT value FROM lock_probe").fetchone()[0] == "uncommitted"
        assert fresh.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_message_stores_keep_live_sidecars_across_external_reader(tmp_path, mode):
    db_path = tmp_path / "lcm.db"
    first = MessageStore(db_path)
    second = None
    try:
        first.append("session", {"role": "user", "content": "before reader"})
        first.commit()
        for artifact in _sqlite_artifacts(db_path):
            if artifact.exists():
                artifact.chmod(mode)
        second = MessageStore(db_path)
        identities = {
            suffix: (tmp_path / ("lcm.db" + suffix)).stat().st_ino
            for suffix in ("-wal", "-shm")
        }
        probe = subprocess.run(
            [sys.executable, "-c", """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
try:
    assert conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 1
finally:
    conn.close()
""", str(db_path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        assert not probe.stderr
        for suffix, inode in identities.items():
            assert (tmp_path / ("lcm.db" + suffix)).stat().st_ino == inode
        first.append("session", {"role": "user", "content": "after reader"})
        first.commit()
        second.append("session", {"role": "user", "content": "second store"})
        second.commit()
        assert first.search("reader", limit=10)
    finally:
        if second is not None:
            second.close()
        first.close()
    with sqlite3.connect(db_path) as fresh:
        assert fresh.execute("SELECT count(*) FROM messages").fetchone()[0] == 3
        assert fresh.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("operation", ["prepare", "restrict"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_private_artifact_rejects_swap_after_parent_stat(
    tmp_path, monkeypatch, operation, link_kind,
):
    db_path = tmp_path / "lcm.db"
    db_path.touch(mode=0o600)
    db_path.chmod(0o600)
    target = tmp_path / "unrelated"
    target.write_bytes(b"untouched")
    target.chmod(0o644)
    real_stat = os.stat
    swapped = False

    def swap_after_stat(path, *args, **kwargs):
        nonlocal swapped
        result = real_stat(path, *args, **kwargs)
        if not swapped and path == db_path.name and kwargs.get("dir_fd") is not None:
            swapped = True
            db_path.unlink()
            if link_kind == "symlink":
                db_path.symlink_to(target)
            else:
                os.link(target, db_path)
        return result

    monkeypatch.setattr(sqlite_util_module.os, "stat", swap_after_stat)
    with pytest.raises(OSError, match="SQLite artifact"):
        if operation == "prepare":
            sqlite_util_module._prepare_private_sqlite_file(db_path)
        else:
            sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
    assert swapped
    assert _mode(target) == 0o644
    assert target.read_bytes() == b"untouched"


@pytest.mark.parametrize("failure", ["timeout", "launch", "null", "empty", "bad_result", "bad_error"])
def test_permission_worker_failure_preserves_live_lock(tmp_path, monkeypatch, failure):
    db_path = tmp_path / "lcm.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(value TEXT)")
        conn.execute("BEGIN IMMEDIATE")
        db_path.chmod(0o644)
        with monkeypatch.context() as patch:
            def fail_worker(*args, **kwargs):
                if failure == "timeout":
                    raise subprocess.TimeoutExpired("permission worker", 10)
                if failure == "launch":
                    raise FileNotFoundError("worker interpreter missing")
                output = {
                    "null": "null", "empty": "{}",
                    "bad_result": '{"result": "true"}',
                    "bad_error": '{"error": []}',
                }[failure]
                return subprocess.CompletedProcess(args[0], 0, output, "")
            patch.setattr(sqlite_util_module.subprocess, "run", fail_worker)
            with pytest.raises(OSError, match="permission worker failed"):
                sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert _mode(db_path) == 0o644
        _assert_writer_blocked(db_path)
    finally:
        conn.close()


def test_permission_worker_rejects_replacement_before_start(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    db_path.write_bytes(b"original")
    db_path.chmod(0o644)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o644)
    real_worker = sqlite_util_module._run_sqlite_permission_worker

    def replace_then_run(path, **kwargs):
        db_path.unlink()
        replacement.rename(db_path)
        return real_worker(path, **kwargs)

    monkeypatch.setattr(sqlite_util_module, "_run_sqlite_permission_worker", replace_then_run)
    with pytest.raises(OSError, match="directory entry changed before permission worker"):
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
    assert _mode(db_path) == 0o644


def test_message_store_creates_private_database_and_sidecars_under_umask_022(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"

    with _process_umask(0o022):
        store = MessageStore(db_path)
        try:
            store.append("session", {"role": "user", "content": "private"})
            store.commit()

            assert _mode(db_path.parent) == 0o700
            assert db_path.with_name(db_path.name + "-wal").exists()
            assert db_path.with_name(db_path.name + "-shm").exists()
            _assert_private_sqlite_artifacts(db_path)
        finally:
            store.close()


def test_message_store_refuses_created_directory_swap_before_chmod(tmp_path, monkeypatch):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o777)
    shared_parent.chmod(0o777)
    db_dir = shared_parent / "database"
    db_path = db_dir / "lcm.db"
    displaced_dir = shared_parent / "database-displaced"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    real_open = os.open
    real_path_chmod = Path.chmod
    swapped = False

    def swap_created_directory():
        nonlocal swapped
        if swapped or not db_dir.is_dir():
            return
        swapped = True
        db_dir.rename(displaced_dir)
        db_dir.symlink_to(unrelated_target, target_is_directory=True)

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path == db_dir.name:
            swap_created_directory()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def swapping_path_chmod(path, mode, *, follow_symlinks=True):
        if path == db_dir:
            swap_created_directory()
        return real_path_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(Path, "chmod", swapping_path_chmod)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(unrelated_target) == 0o755


def test_message_store_tightens_compatible_existing_database_artifacts(tmp_path):
    db_dir = tmp_path / "existing"
    db_dir.mkdir(mode=0o755)
    db_dir.chmod(0o755)
    db_path = db_dir / "lcm.db"
    existing = sqlite3.connect(db_path)
    try:
        existing.execute("PRAGMA journal_mode=WAL")
        existing.execute("CREATE TABLE legacy (value TEXT)")
        existing.execute("INSERT INTO legacy VALUES ('retained')")
        existing.commit()

        wal_path = db_path.with_name(db_path.name + "-wal")
        shm_path = db_path.with_name(db_path.name + "-shm")
        assert wal_path.exists()
        assert shm_path.exists()
        for artifact in (db_path, wal_path, shm_path):
            artifact.chmod(0o644)

        with _process_umask(0o022):
            store = MessageStore(db_path)
            try:
                assert store.connection.execute("SELECT value FROM legacy").fetchone()[0] == "retained"
                assert _mode(db_dir) == 0o755
                _assert_private_sqlite_artifacts(db_path)
            finally:
                store.close()
    finally:
        existing.close()


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_symlinked_sidecar_before_chmod(tmp_path, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    db_path.with_name(db_path.name + suffix).symlink_to(target)

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_hardlinked_sidecar_before_chmod(tmp_path, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    os.link(target, db_path.with_name(db_path.name + suffix))

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.fixture
def inline_permission_worker(monkeypatch):
    # Race injection must run inside the worker's fd-based operation. Keep it
    # in-process for these deterministic security tests only; the lock tests
    # above always exercise the real subprocess boundary.
    monkeypatch.setattr(
        sqlite_util_module, "_run_sqlite_permission_worker",
        sqlite_util_module._chmod_sqlite_artifact_in_worker_at,
    )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_message_store_refuses_sidecar_link_swap_before_chmod(
    tmp_path,
    monkeypatch,
    inline_permission_worker,
    link_kind,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-wal")
    sidecar.write_text("replace me", encoding="utf-8")
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            if link_kind == "symlink":
                sidecar.symlink_to(target)
            else:
                os.link(target, sidecar)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(target) == 0o644


def test_message_store_refuses_sidecar_replacement_after_open_before_fstat(
    tmp_path,
    monkeypatch,
    inline_permission_worker,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-journal")
    replacement = tmp_path / "replacement-journal"
    sidecar.write_bytes(b"replace me")
    replacement.write_bytes(b"unrelated")
    sidecar.chmod(0o644)
    replacement.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            replacement.rename(sidecar)
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError, match="directory entry changed while opening"):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(sidecar) == 0o644


def test_message_store_tolerates_sidecar_disappearing_between_stat_and_open(
    tmp_path,
    monkeypatch,
    inline_permission_worker,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    disappeared = False

    def disappearing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal disappeared
        if not disappeared and dir_fd is not None and path == journal.name:
            disappeared = True
            journal.unlink()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", disappearing_open)

    store = MessageStore(db_path)
    try:
        assert disappeared is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_tolerates_sidecar_unlinked_between_open_and_fstat(
    tmp_path,
    monkeypatch,
    inline_permission_worker,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    unlinked = False

    def unlinking_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal unlinked
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not unlinked and path == journal.name:
            unlinked = True
            journal.unlink()
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", unlinking_open)

    store = MessageStore(db_path)
    try:
        assert unlinked is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_preserves_sqlite_memory_sentinel_and_cwd_permissions(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    store = MessageStore(":memory:")
    try:
        store.append("session", {"role": "user", "content": "memory-only"})
        store.commit()
        assert store.connection.execute("SELECT content FROM messages").fetchone()[0] == "memory-only"
    finally:
        store.close()

    assert _mode(tmp_path) == 0o755
    assert not (tmp_path / ":memory:").exists()


def test_maintenance_creates_private_backups_and_tightens_existing_slot(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    backup_dir.chmod(0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    rotate_path.write_bytes(b"stale")
    rotate_sidecars = [
        rotate_path.with_name(rotate_path.name + suffix)
        for suffix in _SQLITE_SIDECAR_SUFFIXES
    ]
    for artifact in [rotate_path, *rotate_sidecars]:
        if not artifact.exists():
            artifact.write_bytes(b"stale")
        artifact.chmod(0o644)

    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        with _process_umask(0o022):
            timestamped = backup_database(engine)
            rotated = rotate_backup_database(engine)

        assert timestamped["ok"] is True
        assert rotated["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(timestamped["backup_path"])
        _assert_private_sqlite_artifacts(rotate_path)
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()

        for backup_path in (timestamped["backup_path"], rotate_path):
            with sqlite3.connect(backup_path) as restored:
                assert restored.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                assert restored.execute("SELECT content FROM messages").fetchone()[0] == "backup"
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_backups_work_without_fchmod(tmp_path, monkeypatch, operation):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()
    backup_dir = tmp_path / "backups"
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    if hasattr(maintenance_module, "_FCHMOD"):
        monkeypatch.setattr(maintenance_module, "_FCHMOD", None)
    else:  # RED control for the rejected head, which read os.fchmod directly.
        class _OSWithoutFchmod:
            fchmod = None

            def __getattr__(self, name):
                return getattr(os, name)

        monkeypatch.setattr(maintenance_module, "os", _OSWithoutFchmod())
    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(result["backup_path"])
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_rejects_symlinked_backup_directory_before_chmod(
    tmp_path,
    operation,
):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_parent = tmp_path / "backups"
    backup_parent.mkdir()
    backup_dir = backup_parent / "lcm"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    backup_dir.symlink_to(unrelated_target, target_is_directory=True)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is False
        assert _mode(unrelated_target) == 0o755
        assert list(unrelated_target.iterdir()) == []
    finally:
        store.close()


def test_rotate_backup_failure_preserves_existing_atomic_slot(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    previous_backup = b"known-good-backup"
    rotate_path.write_bytes(previous_backup)
    rotate_path.chmod(0o644)
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        rotate_backup_path=lambda: rotate_path,
    )

    def fail_backup(_destination):
        raise sqlite3.OperationalError("synthetic backup failure")

    monkeypatch.setattr(store, "backup", fail_backup)
    try:
        with _process_umask(0o022):
            result = rotate_backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic backup failure"
        assert rotate_path.read_bytes() == previous_backup
        assert _mode(backup_dir) == 0o700
        assert _mode(rotate_path) == 0o600
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()
    finally:
        store.close()


def test_timestamped_backup_flush_failure_leaves_no_empty_artifact(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    backup_dir = tmp_path / "backups"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
    )

    def fail_flush(_engine):
        raise sqlite3.OperationalError("synthetic flush failure")

    monkeypatch.setattr(maintenance_module, "flush_engine_connections", fail_flush)
    try:
        result = backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic flush failure"
        assert list(backup_dir.glob("*.sqlite3")) == []
    finally:
        store.close()
