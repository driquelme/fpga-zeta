# Debugging TODO

Tracking file for temporary debug prints/traces (project convention).

Entry format — add one entry per print/trace when introduced; when removed from source,
do NOT delete the entry: mark it `removed`, and note whether it was actually useful.

```
## <relative/path/to/file>:<line> — <status: active|removed>
- Session: <claude session id>
- Context: <what was being debugged>
- Goal: <what the print/trace should reveal>
- Remove when: <criteria>
- Useful?: <filled in at removal>
```

_No active entries._
- tb/zeta/test_os_grid_sum.py:~92 (mismatch loop with cocotb.log.error), session
  6582edeb-bf76-4eb8-8ab4-4ba715f6d400 (M21): localize RTL-vs-golden divergence in
  os_grid_sum (which points/components differ). Remove once bit-exact at Z64/Z128.
  - REMOVED from source 2026-07-13. Useful: yes — the "all points wrong, right
    magnitude" pattern pointed at the sweep front-end, which exposed a stale
    lnn_q register feeding both fx_mul_mod1 instances (fixed by wiring the
    live entry input).
