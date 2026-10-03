# Recording what people measure

Every time someone measures enrollment on the site, the server keeps one line of text for that animal
and, if you give it an address, sends the line to you. One line per distinct animal.

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

## Where it goes

1. `web/records/readings.jsonl` on the server, always (git-ignored). `TRILO_RECORD_DIR` moves it. In Docker,
   mount that folder or the file is lost when the container is replaced.
2. Your own address, if set:

```
TRILO_RECORD_URL=https://...        # each new line is POSTed here as JSON
TRILO_RECORD_TOKEN=some-long-secret # sent as "Authorization: Bearer ..." (optional)
TRILO_RECORD=0                      # turns recording off entirely
```

If the address is unreachable the line stays on the server and is retried with the next measurement and
when the server starts. `web/records/sent.txt` lists what has arrived; `web/records/errors.log` says why something has not.

## A private receiver: a Google Sheet only you can open

1. Make a new Google Sheet. Extensions -> Apps Script, paste this, and put your own secret in `SECRET`:

```javascript
const SECRET = 'some-long-secret';
function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  if (d.secret !== SECRET) return ContentService.createTextOutput('no');
  const sh = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  if (sh.getLastRow() === 0) sh.appendRow(['time', 'hash', 'schema', 'instrument', 'limited_by', 'class', 'theta_joint_deg', 's_tail', 'params', 'reading']);
  const r = d.reading || {};
  sh.appendRow([d.time, d.hash, d.schema, d.instrument, r.limited_by, r.enroll_class, r.theta_joint_deg, r.s_tail, JSON.stringify(d.params), JSON.stringify(r)]);
  return ContentService.createTextOutput('ok');
}
```

2. Deploy -> New deployment -> Web app, execute as you, access "Anyone". Copy the web app URL.
3. On the server: `TRILO_RECORD_URL=<that URL>` and `TRILO_RECORD_SECRET=some-long-secret`.

Apps Script cannot read request headers, so for this receiver the secret travels in the body:
when `TRILO_RECORD_SECRET` is set the server adds `"secret": "..."` to what it sends (never to the stored file).
The sheet stays private to your Google account; the URL only accepts lines that carry the secret.

Any other receiver that accepts a JSON POST works the same way (a small server of your own, a webhook).
