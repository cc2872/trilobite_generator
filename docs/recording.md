# Recording what people measure

Every time someone measures enrollment on the site, the server keeps one line of text for that animal
and, if you give it an address, sends the line to you. One line per distinct animal.

There are two lists, both append-only JSONL in `web/records/` (git-ignored), both sent on if an address is set:

- **The clean list** — `readings.jsonl`, one line per *distinct measured animal* (`kind: "reading"`). This is the dataset.
- **The raw list** — `agent_log.jsonl`, one line per *agent request* to `/api/agent/make` (`kind: "agent_request"`),
  with **no dedup**: repeats, rejected calls (bad preset/parameter, missing token, server busy) and errors are all kept.
  It is the audit trail of who asked the robot for what. The clean list is a subset (the ones that measured).

The `kind` field tells a receiver which list a line belongs to, so one Sheet can keep both on separate tabs (below).

## What a line holds

```json
{"key": "3f2a9c11d0@6.2@2.1", "hash": "3f2a9c11d0", "schema": "6.2", "instrument": "2.1",
 "time": "2026-10-03T21:55:02Z", "source": "site",
 "text": "3f2a9c11d0  schema 6.2  instrument 2.1  closed / double  theta/joint 22.81  s_tail 0.258",
 "params": { ...the full parameter table... },
 "reading": { ...everything the instrument returned, minus the animation data... }}
```

- **Distinct** means the parameter hash together with the schema and instrument versions. The same animal
  measured twice is stored once; the same animal under a new instrument version is stored again.
- **Nothing about the visitor** is stored or sent: no address, no browser, no cookie.
- Only `/api/measure` (the pin instrument, "Curl it") is recorded. Building or viewing without measuring is not.

## The raw list: every agent request

A raw-list line (`agent_log.jsonl`, `kind: "agent_request"`) holds the time, the request (preset, chars, the
parameters set, whether a build was asked for), the `outcome` (`done`, `running`, `unknown_preset`, `unknown_chars`,
`unknown_params`, `unauthorized`, `busy`, `error`), the HTTP `code`, the resulting `hash` when there is one, and a
one-line `summary` when it measured. No dedup: ask for the same animal ten times and you get ten lines.

**Live view:** `GET /api/agent/log` returns the raw list — JSON (newest first), or a small self-refreshing HTML
table in a browser (or with `?format=html`). `?limit=N` caps how many lines come back (default 200, max 2000).
It is gated by the **agent token** (the same `?token=` / `Authorization: Bearer` as `make`), so only you can read it.

## Where it goes

1. `web/records/readings.jsonl` and `web/records/agent_log.jsonl` on the server, always (git-ignored).
   `TRILO_RECORD_DIR` moves the folder. In Docker, mount it or both files are lost when the container is replaced.
2. Your own address, if set — every new line from *either* file is POSTed, carrying its `kind`:

```
TRILO_RECORD_URL=https://...        # each new line is POSTed here as JSON
TRILO_RECORD_TOKEN=some-long-secret # sent as "Authorization: Bearer ..." (optional)
TRILO_RECORD=0                      # turns recording off entirely
```

If the address is unreachable the line stays on the server and is retried with the next measurement and
when the server starts. `web/records/sent.txt` lists what has arrived; `web/records/errors.log` says why something has not.

## A private receiver: a Google Sheet only you can open

1. Make a new Google Sheet. Extensions -> Apps Script, paste this, and put your own secret in `SECRET`:

The script routes by `kind` so the clean list and the raw list land on their own tabs (`readings` and `agent_log`):

```javascript
const SECRET = 'some-long-secret';
function tab(name, header) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sh = ss.getSheetByName(name) || ss.insertSheet(name);
  if (sh.getLastRow() === 0) sh.appendRow(header);
  return sh;
}
function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  if (d.secret !== SECRET) return ContentService.createTextOutput('no');
  if (d.kind === 'agent_request') {
    const q = d.request || {};
    tab('agent_log', ['time', 'outcome', 'code', 'preset', 'chars', 'set', 'build', 'hash', 'summary'])
      .appendRow([d.time, d.outcome, d.code, q.preset, (q.chars || []).join(','), JSON.stringify(q.set || {}), q.build, d.hash || '', d.summary || '']);
  } else {
    const r = d.reading || {};
    tab('readings', ['time', 'hash', 'schema', 'instrument', 'limited_by', 'class', 'theta_joint_deg', 's_tail', 'params', 'reading'])
      .appendRow([d.time, d.hash, d.schema, d.instrument, r.limited_by, r.enroll_class, r.theta_joint_deg, r.s_tail, JSON.stringify(d.params), JSON.stringify(r)]);
  }
  return ContentService.createTextOutput('ok');
}
```

2. Deploy -> New deployment -> Web app, execute as you, access "Anyone". Copy the web app URL.
3. On the server: `TRILO_RECORD_URL=<that URL>` and `TRILO_RECORD_SECRET=some-long-secret`.

Apps Script cannot read request headers, so for this receiver the secret travels in the body:
when `TRILO_RECORD_SECRET` is set the server adds `"secret": "..."` to what it sends (never to the stored file).
The sheet stays private to your Google account; the URL only accepts lines that carry the secret. An older script
that ignores `kind` still works — every line, raw or clean, just lands on the first sheet.

Any other receiver that accepts a JSON POST works the same way (a small server of your own, a webhook).
