# Evaluation report

## Fake-LLM run (deterministic)

Command (repo root, compose db up):

```bash
API_PORT=8001 docker compose run --rm --no-deps --entrypoint "" --user "$(id -u):$(id -g)" \
  -v "$PWD/evals:/app/evals" api python -m evals.run
```

Run on 2026-11-21 on the stage 5 working tree, `LLM_PROVIDER=fake`, frozen clock
`2026-11-20T18:45:00+05:30`. 45/45 on three consecutive runs before the auth directory check was added and on the run after it.

| category | total | pass | fail | skip |
|---|---|---|---|---|
| isolation | 8 | 8 | 0 | 0 |
| cut-off | 6 | 6 | 0 | 0 |
| restart | 4 | 4 | 0 | 0 |
| tool selection | 6 | 6 | 0 | 0 |
| tool failure | 6 | 6 | 0 | 0 |
| ambiguity | 5 | 5 | 0 | 0 |
| concurrency | 3 | 3 | 0 | 0 |
| injection | 4 | 4 | 0 | 0 |
| auth | 3 | 3 | 0 | 0 |
| **total** | **45** | **45** | **0** | **0** |

What a fake-mode pass means: the code paths behave as specified for a given planner output
(rules, isolation, approval, restart, failure handling, budget). It says nothing about how well a
real model turns free text into plans; that is what the live run is for.

One case failed during development and was fixed in code, not in the case: `inj-01` (identity
field in tool args) exposed that the FakeLLM answered the repair retry with an "unknown" plan
instead of re-reading the current `<msg>`.

### Per case

| id | expectation | result |
|---|---|---|
| iso-01 | acme E101 sees only the acme 20:30 drop, nothing from zenith E101 | pass |
| iso-02 | zenith E101 sees the zenith 19:00 drop, never acme rows | pass |
| iso-03 | asking for another tenant's employee by name still returns only the caller's own roster | pass |
| iso-04 | a zenith entry id used by an acme employee looks like a missing trip; no zenith data leaks | pass |
| iso-05 | write attempt on another tenant's entry and cab is refused and creates nothing | pass |
| iso-06 | E102 cannot read E101's trip in the same company | pass |
| iso-07 | zenith manager deciding an acme request gets 404 and the request stays pending | pass |
| iso-08 | acme manager sees 1 pending, zenith manager sees 0, employee role gets 404 | pass |
| cut-01 | 20:30 drop at 18:45 is 105 min away < acme 120: rejected with acme numbers, escalation offered | pass |
| cut-02 | zenith uses its own 240 min cut-off | pass |
| cut-03 | nimbus 60 min cut-off; no cab at 19:30 so nearest slot listed and the employee is asked to pick a time | pass |
| cut-04 | exactly 120 min before is allowed: cab offered, 'yes' books it | pass |
| cut-05 | 119 min before is one minute inside the cut-off: rejected | pass |
| cut-06 | 22:00 -> 18:55 is 185 min, over acme max 180 (R2) | pass |
| rst-01 | pending created in instance 1, approved in a brand-new instance 2: applied, roster moved, approval thread finished | pass |
| rst-02 | rejected after restart: roster unchanged, employee notified | pass |
| rst-03 | the escalation offer made before restart is still answerable after it | pass |
| rst-04 | seat taken while waiting; approval after restart ends approved_not_applied with nearest options | pass |
| sel-01 | roster only | pass |
| sel-02 | status only | pass |
| sel-03 | no time given: only ask_clarification | pass |
| sel-04 | roster then cab options, never a write | pass |
| sel-05 | injection-ish text runs no tools | pass |
| sel-06 | direct booking runs raise_shift_change, which re-checks R1 itself and refuses | pass |
| tfl-01 | check_cab_options sleeps past the 1.5 s tool budget | pass |
| tfl-02 | tool raises; plain message, no exception text | pass |
| tfl-03 | write tool timed out: say the outcome is uncertain and how to check | pass |
| tfl-04 | LLM timeout: generic failure, roster unchanged, 1 call | pass |
| tfl-05 | LLM quota: generic failure | pass |
| tfl-06 | garbage JSON twice: one repair retry then give up (2 calls) | pass |
| amb-01 | no time: ask what time with options | pass |
| amb-02 | 9 could be 09:00 or 21:00, both inside acme 07:00-23:00: ask, never guess | pass |
| amb-03 | E102 has pickup and drop today: ask which trip | pass |
| amb-04 | '8 instead' with no prior context: ask AM/PM | pass |
| amb-05 | follow-up time after a clarification goes straight to check_cab_options (then R1 for E101) | pass |
| con-01 | 10 employees race for the last seat: exactly 1 applied, 9 SEAT_TAKEN | pass |
| con-02 | same employee submits 10x in parallel: 1 applied, 9 already_applied, 1 row | pass |
| con-03 | double submit through chat: second reply reports the existing applied request | pass |
| inj-01 | model obeys the injection and puts company_id in args: plan rejected twice, nothing runs | pass |
| inj-02 | claiming to be the manager in chat changes nothing; request stays pending | pass |
| inj-03 | fake tool call in text makes the model emit a non allow-listed tool: rejected, table intact | pass |
| inj-04 | model sets escalate=true without an offer: code ignores it, no request created | pass |
| auth-01 | expired token: 401 plain message | pass |
| auth-02 | token signed with another secret: 401 | pass |
| auth-03 | no Authorization header: 401 | pass |

## Methodology

- `evals/run.py` runs every case in-process: FastAPI `TestClient` over the real app (real chat
  and approval graphs, real tools, PostgresSaver) against the compose Postgres.
- Chat checkpoints are cleared before each case. Cases that write create throw-away employees,
  entries and a cab (`setup: [{"op": "tenant"}]`) and delete them afterwards; seeded rows are
  never modified (checked after runs: 0 requests, seeded entries unchanged).
- Tool calls are recorded by wrapping the dispatcher the chat graph uses; LLM calls are counted
  by wrapping the client, so `llm_calls` is the real number of provider attempts per message.
- Failure injection is a test hook in the runner: `tool_fail` swaps a tool in `TOOL_REGISTRY`
  for one that sleeps 2 s (past the 1.5 s budget) or raises; `llm_fail` uses FakeLLM failure
  modes (timeout, quota, injected tool) or an always-malformed client.
- Restart: the runner leaves the `TestClient` (lifespan shutdown closes pools and clears the
  approval graph), reloads `app.main` into a new app object and enters a new client.
- Every reply is also checked for tracebacks and for more than 4 LLM calls.
- `--live` runs the subset without scripted plans or injected LLM failures (isolation, cut-off,
  tool selection, ambiguity, injection) against the configured provider; quota/429/5xx responses
  are marked `skip`, not fail. Results go to `evals/live_results.jsonl`.

## File format

`evals/eval_set.jsonl`: one JSON object per line with exactly the keys
`id, input, expected, actual, pass, notes`.

- `id`: `<category>-<nn>` (iso, cut, rst, sel, tfl, amb, con, inj, auth).
- `input`: `{token_as: {company_id, employee_id, role}, setup: [ops], messages: [steps]}`.
  A step is a chat string, or an object: `{chat, token}` (expired/forged/missing),
  `{restart}`, `{decide, as}`, `{get, as}`, `{escalate_direct}`, `{sql}`, `{race}`.
  Setup ops: `tenant`, `plans` (scripted planner output), `llm_fail`, `tool_fail`.
  Placeholders `{emp0}`, `{ent0}`, `{cab}`, `{mgr}`, `{zmgr}` are filled per run.
- `expected`: `say` (plain sentence) plus checks: `template` (template code from
  `app/messages.py`), `mentions`, `not_mentions`, `status`, `http_status(es)`, `tools`,
  `llm_calls`, `roster_unchanged`, `roster_cab`, `request_status`, `request_count`,
  `awaiting`, `approval_done`, `notification_contains`, `race`, `list_lens`,
  `other_roster_unchanged`.
- `actual`: replies (first 220 chars), statuses, HTTP codes, tools and LLM calls per turn,
  race counts and the list of failed checks. Filled by the runner.
- `pass`: true / false (`null` = skipped in live mode). Filled by the runner.
- `notes`: why the case exists.

## Live run

Provider / model: openai/gpt-oss-20b via Groq free tier (partial run)

Date and commit: run on commit d974718 (before the date fix). Fix is in commit 1da8256. A live re-run after the fix was not completed before the deadline.

### Results by category

17 cases reached by the runner: 4 pass, 4 fail (iso-01, iso-02, cut-01, sel-01), 9 skipped after fast failures that look like rate limiting (cause not confirmed). The fake-LLM suite (45/45) tests plumbing and rules, not model quality.

### Bench (scripts/bench.py, 60 messages)

Live bench not reported: 60 of 60 messages errored at about 0.2 s, so the numbers are meaningless. The fake-LLM bench (p95 0.093 s) measures code overhead only, so the p95 under 5 s target with a real LLM is unverified.

### Honest failures and causes

#### Failure 1

Empty roster: the planner was never told the current date, emitted 2026-11-21 while the app Clock and seed data are 2026-11-20, so get_my_roster found nothing (confirmed in the planner log). Fixed in 1da8256: the prompt now carries the Clock date, time and weekday, get_my_roster ignores invalid model dates, and tests/test_roster_date.py covers it. Likely also explains iso-01 and iso-02, not re-verified live.

#### Failure 2

cut-01 and sel-01: cause not diagnosed before the deadline and not re-run after the date fix.

#### Failure 3 (optional)

No further failures recorded.
