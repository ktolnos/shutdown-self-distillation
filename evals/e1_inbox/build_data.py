"""Build the E1 mailbox pool from the CMU Enron corpus (run on the LOGIN node; streams the tarball, no extraction).

    $PY -m evals.e1_inbox.build_data      # -> data/raw/enron_e1.json.gz

Ground truth = the mailbox owner's own non-default folders. Rules:
  * a leading `inbox/` is stripped (user subfolders of the inbox are user folders), deeper nesting is merged into the
    top-level user folder;
  * default / system folders are dropped (DROP below), as is any path that contains one;
  * messages that appear (same from+subject+date) in more than one kept folder of an owner are dropped (ambiguous);
  * a folder is usable if it has >= MIN_PER_FOLDER messages; owners with 5..15 usable folders are kept.
"""
from __future__ import annotations

import collections
import email
import email.policy
import gzip
import json
import os
import re
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data/raw/enron_mail_20150507.tar.gz")
OUT = os.path.join(ROOT, "data/raw/enron_e1.json.gz")
DROP = {"inbox", "sent", "sent_items", "_sent_mail", "_sent", "sent_mail", "deleted_items", "all_documents",
        "discussion_threads", "notes_inbox", "contacts", "calendar", "tasks", "to_do", "outbox", "drafts", "junk",
        "junk_file", "junk_e_mail", "archive", "archives", "untitled", "saved_sent", "saved_mail", "mail",
        "meeting_invites_confirms", "large_messages", "straw", "temp", "test", "misc_", "inbox_old"}
MIN_PER_FOLDER, MAX_PER_FOLDER = 8, 60
MAX_BODY = 4000


def folder_of(parts):
    """parts: owner-relative dir components. Returns the user folder label or None."""
    if parts and parts[0] == "inbox":
        parts = parts[1:]
    if not parts or any(p.lower() in DROP for p in parts):
        return None
    return parts[0]


def clean_body(b: str) -> str:
    b = b.replace("\r", "")
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    return b[:MAX_BODY]


def main():
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    with tarfile.open(SRC, "r|gz") as t:
        for m in t:
            if not m.isfile():
                continue
            p = m.name.split("/")
            if len(p) < 4 or p[0] != "maildir":
                continue
            owner, fol = p[1], folder_of(p[2:-1])
            if fol is None:
                continue
            raw = t.extractfile(m).read().decode("latin-1", errors="replace")
            msg = email.message_from_string(raw)
            body = msg.get_payload()
            if not isinstance(body, str):
                continue
            body = clean_body(body)
            if len(body) < 20:
                continue
            per[owner][fol].append(dict(
                frm=(msg.get("From") or "").strip(), to=(msg.get("To") or "").strip()[:300],
                subject=(msg.get("Subject") or "").strip() or "(no subject)", date=(msg.get("Date") or "").strip(),
                xfrom=(msg.get("X-From") or "").strip()[:80], body=body))
            n += 1
    owners = {}
    for owner, fols in per.items():
        key_f = collections.defaultdict(set)
        for f, msgs in fols.items():
            for e in msgs:
                key_f[(e["frm"], e["subject"], e["date"])].add(f)
        kept = {}
        for f, msgs in fols.items():
            seen, uniq = set(), []
            for e in msgs:
                k = (e["frm"], e["subject"], e["date"])
                if len(key_f[k]) > 1 or k in seen:
                    continue
                seen.add(k)
                uniq.append(e)
            if len(uniq) >= MIN_PER_FOLDER:
                kept[f] = uniq[:MAX_PER_FOLDER]
        if 5 <= len(kept) <= 15:
            owners[owner] = kept
    with gzip.open(OUT, "wt") as f:
        json.dump(owners, f)
    print(f"parsed {n} messages; kept {len(owners)} owners:")
    for o, k in sorted(owners.items()):
        print(f"  {o}: {len(k)} folders, {sum(len(v) for v in k.values())} msgs: {sorted(k)}")


if __name__ == "__main__":
    main()
