# Handoff prompt

Prepare a sanitized handoff for task `TNN` containing:

- completed outcome and acceptance evidence;
- files and public artifacts changed;
- exact verification performed;
- important decisions and ADRs;
- known limitations or blockers;
- downstream tasks newly unblocked;
- restart commands and required artifact hashes.

Exclude raw transcripts, private agent session IDs, credentials, account details,
and unsanitized cloud URLs.
