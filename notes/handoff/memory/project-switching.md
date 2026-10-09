---
name: project-switching
description: To switch projects, hand off to a new session started in the target directory; never cd mid-session
metadata:
  type: feedback
---

When the user switches projects, do not `cd` or retarget the current session into the new repo. Instead write a handoff file in the new repo and copy the relevant memories into the new project's memory dir (`~/.claude/projects/<dir-with-slashes-as-dashes>/memory/`). Then have the user start a fresh `claude` (or `claude remote-control`) in that directory.

**Why:** session transcripts and memory are keyed to the directory a session *started* in. The 2026-10-07 shutdown session started in the OPD repo, so its transcript and memory stayed there: it could not be resumed from the new repo, and its last message showed in the UI of a different session without reaching the model.

**How to apply:** for any "switch active project" request, follow [[project-overview]]'s pattern: handoff file in `notes/handoff/`, memory copied over, new session started in the target dir.
