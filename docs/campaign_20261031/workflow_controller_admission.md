# Controller process admission

The controller now starts a waiting worker before the registered payload. This
closes the crash window between process creation and durable PID registration.
The parent records the attempt and one-use token, starts the worker, commits its
PID and creation time, then sends the token. Missing, stale or timed-out
admission cannot execute payload code. The OS integration lease is unchanged.

Admission uses a separate anonymous pipe, so payload stdin remains inherited.
Windows inherits only the read handle explicitly; POSIX uses `pass_fds`. The
controller retains the write handle. An absolute worker script path preserves
arbitrary registered working directories. Payload argv, cwd, merged logs, exit
status and configured retry policy retain their existing meanings.

Windows virtual-environment redirectors can create another Python process. The
worker accepts that relationship only when the registered launcher is its live
immediate parent with matching creation time. Before payload launch it
transactionally rebinds the job to its actual PID and creation time.

After admission the worker records `LAUNCHING` before spawning the payload and
`PAYLOAD` with its observed identity. Controller-pipe EOF stops the payload and
observed descendants. The worker remains represented by its durable identity
until cleanup finishes, then records `FINISHED`. Creation identities are checked
before process termination.

If controller and worker both disappear during an ambiguous admitted launch,
the job becomes `RECOVERY_REQUIRED`, which returns a failing CLI status. It is
not automatically requeued while an unverified payload may exist. Preserve the
database, logs and outputs for inspection. Arbitrary detached daemons cannot be
reconstructed automatically after every supervising process is lost.

Synthetic tests exercise repeated deaths before PID registration and before
token release, stale identity/token, timeout/EOF, failed PID writes, inherited
stdin, exit/log behavior, and owner death around launch and during a payload
with a grandchild. No corpus or model execution is required.
