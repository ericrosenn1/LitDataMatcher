# Recovery, continuation and safe return paths

## Read before restarting

Read the compact state, next action, build specification digest, actual Git status, common job/lease store and current process identities. Verify the source/config/model versions and artifact hashes. Do not delete outputs because a previous chat ended or begin another full baseline audit.

An interrupted model conversation and an interrupted data job are different. A healthy task-owned data job may still be running. Acquire only the necessary role/lease, collect completed outputs first, and requeue truly abandoned jobs. Never start two leads or overwrite concurrent work.

## Reuse validated work

Cache acquisition by source identity/version/hash. Cache parsed records by parser/config and input. Cache inference by text/model/tokenizer/prompt/schema/generation settings. Cache downstream matches by requirement/capability/evidence versions. Changed sources or scientific interpretation invalidate affected descendants, not unrelated downloads and profiles.

Validate file contents/schema/checksum before reusing them. Write to a temporary artifact and atomically promote only after success. Keep valid completed partitions and mark failed partitions explicitly. Do not append duplicate records on replay. A partial corpus is usable only with its actual coverage reported.

## Diagnose and repair

A network/API transient receives bounded backoff and Retry-After compliance; authentication does not receive infinite retry. A corrupt file is quarantined and reacquired once through an allowed route. A parser/schema defect gets a failing fixture and a focused repair. OOM reduces task-owned batches/concurrency; it is not a reason to change scientific definitions. A lock conflict defers the writer and rechecks ownership. A quota limit checkpoints and waits without hidden paid fallback.

Repair one cause and rerun affected validation. If the same signature returns without new evidence, try a materially different justified fix or mark the precise blocker and move to independent work. Do not keep asking agents to 'try harder' against an identical failure. A whole-experiment restart needs demonstrated global invalidation.

Kill only the verified process tree belonging to the failed task. Record PID creation time and parent lineage to avoid PID reuse. Preserve numeric exit status and final logs. Never run a blanket kill of all Python/Node/WSL/Codex processes.

## Pauses and approvals

A user pause or cancellation must persist in the live store and task snapshot. Supervisor checks respect it and never resume automatically until the user's later instruction. Scheduling and provider reset waits are distinct from manual pause.

When required access/spending/source authority/scientific scope is unresolved, state exactly what is missing, what is safely preserved, and which tasks can continue. Continue those tasks. No generic 'cannot proceed' when a nonessential donor failed.

## Context handoff

Before an orderly session boundary, update state and exact next action, checkpoint source and compact evidence, push when permitted, and release/transfer the lead lease using the verified controller. Record currently running owned jobs rather than assuming they ended. Keep one current handoff view, not dozens of copied folders.

If no supported unattended resumption control exists, expose the exact minimal resume instruction and mark automation accordingly. A hypothetical schedule or agent capability is not recovery evidence. Completing the package now does not start a background build.
