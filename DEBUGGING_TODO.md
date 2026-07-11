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
