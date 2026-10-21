# SQLite permission changes and live connections

The `ryonakae` branch carries the local SQLite permission fix. Keep `main`
aligned with upstream; do not apply local changes there.

## Why permission changes need isolation

On POSIX, closing a separate descriptor for a file can release record locks
held by SQLite in the same process. In WAL mode, this includes locks on the
`-shm` file. The previous `open` / `fchmod` / `close` sequence could therefore
remove mutual exclusion while another LCM connection was still active.

Permission handling now works as follows:

- Every existing artifact, including mode `0600`, is validated in a fresh
  Python subprocess using no-follow open and descriptor-based checks.
- File creation and permission tightening run in that subprocess too.
  Only the verified parent directory descriptor is explicitly inherited.
- The worker uses the existing no-follow, inode/device identity, and link-count
  checks before changing permissions. Existing file identities are also checked
  across the subprocess boundary.
- A worker failure or timeout fails the operation. There is no fallback that
  opens the artifact in the calling process.

This covers macOS without depending on Linux `O_PATH`. It also keeps
fd-based validation when tightening loose permissions, rather than replacing
it with a pathname-based chmod susceptible to link replacement.

The timeout is 10 seconds **per artifact**, not per store initialization. A pass
can visit the DB and three sidecars; MessageStore prepares artifacts before
connecting and restricts them again afterward. Successful but slow workers can
therefore accumulate startup latency across those passes. The first failure
aborts the operation. Even private stores pay subprocess startup cost during
construction, but ordinary message writes do not invoke this permission pass.

## Verification

`tests/test_storage_permissions.py` covers active write-lock retention through
an independent process for WAL and DELETE journals, `0600` and `0644` modes,
and both storage preparation and permission restriction. It also covers two
live MessageStores, an external reader closing its connection, sidecar inode
retention, subsequent writes, and database integrity.

Run tests with a disposable `HERMES_HOME`, `LCM_DATABASE_PATH`, and `TMPDIR`.
Do not point diagnostic or test processes at a production database with
unpatched connections still running.

## Deployment is separate from repair

This fix prevents the demonstrated lock release; it does not repair an already
corrupt database or restore locks lost by an older process. All processes using
the database must load the patched code before resuming normal operation.
Preserve damaged data and verify a replacement or recovered database separately.
Gateway lifecycle changes remain an operator action.
