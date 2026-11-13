# ChatCarry

Carry a project from one AI chat to the next without losing what matters.

Long AI chats hit limits, and when you start a new chat you lose the decisions, the latest code and the open bugs. ChatCarry turns a chat into a **project state** you can review, fix and hand to any AI. It transfers the project's working memory, not just the text of the conversation.

## What it does

1. **Get the chat in.** Paste it, upload a `.txt` / `.md`, or import a ChatGPT or Claude data export (`conversations.json`) and pick a conversation from the list.
2. **Extract** the project state: goal, tech stack, related projects/test context, requirements, decisions (with the reason), completed work, open bugs, "do not change" rules, next tasks and the latest code. A progress bar shows which part of a long chat is being read, with a rough time estimate and a Cancel button.
3. **Review** it. Items link to a speaker-labelled turn or an explicitly labelled unlabelled source block. The reference is block/message-level, not an exact quote; long-source excerpts may show only the beginning. Items the extractor could not trace are flagged. Edit, delete or add items.
4. **Hand off.** Copy a ready-to-paste prompt for ChatGPT, Claude, Gemini or any AI, or download it as Markdown.
5. **Save** the project. It appears under **My projects** on the first page. Next time choose **Add a new chat** and the new chat is merged into the same project (version 2, 3, ...). You can keep several projects and open, continue or delete each one.

## What is different from a plain chat exporter

| Feature | How it works |
|---|---|
| Source references | Each extracted item cites a source number (`S1#12`). Clicking one shows a saved excerpt. Unlabelled source blocks are clearly marked as blocks, not reconstructed original turns; long excerpts may not contain the exact supporting passage. Invented citations are dropped and the item is flagged "no source". |
| Replaced decisions | "Use SQLite instead of Firebase" marks the earlier Firebase decision as replaced. It moves to a "do not use" section so the next AI does not follow it. |
| Conflict detector | Two live decisions that pick different options for one topic (Firebase, then MongoDB) are flagged. You pick the winner, or the handoff tells the next AI to ask you. |
| Latest-code tracker | Only the newest version of each file is kept, with a note of how many versions were seen. The code is copied by the app straight from the chat, never retyped by the model, so it cannot be changed or cut short, and Gemini has much less to write (faster). |
| Conservative deduplication | Exact repeats, reordered versions of the same meaningful words, and clearly repeated phrases with extra context can merge. Similar vocabulary alone is not enough; numeric distinctions (`120 bpm` vs `130 bpm`) stay separate. |
| Entity classification | Named applications/projects used as examples or test contexts are kept separate from actual technologies; ambiguous entities are omitted rather than guessed. |
| Evidence-based status | Technologies distinguish approved, proposed, considered, in-use, implemented-but-untested and implemented-and-verified states. Completed work separately distinguishes implementation from successful testing. |
| Privacy filter | API keys, tokens, passwords, private keys, emails and phone numbers are replaced with placeholders before the text is split or sent anywhere. |
| Living projects | Saved projects keep growing: each new chat is merged into the project and the version goes up. |
| Check before you copy | A short checklist (final requirements? latest code? all errors listed?) and a "checked by the user" line in the handoff. |
| Export import | Reads the official ChatGPT and Claude exports in the browser, follows the branch you actually used, and skips tool output and hidden messages. |

## Run it

**Windows, easiest:** double-click `start_chatcarry.bat`. The first time it creates a private Python environment and installs Flask (about a minute, needs internet). After that it starts the app and opens your browser. It needs Python 3.9 or newer from python.org, with "Add python.exe to PATH" ticked.

**Any system:**

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000, click **Load sample chat**, then **Extract project state**.

### Gemini key

For good results use Gemini. Get a free key from Google AI Studio and either paste it into the Settings box on the first page, or put it in a file named `.env` next to `app.py` so you never paste it again:

```
GEMINI_API_KEY=your-key-here
```

The launcher creates `.env` from `.env.example` the first time. No quotes, no spaces. A key in `.env` is read once at start-up and is never sent to the browser or written to a log; the page only learns that a key exists. Do not share or upload the `.env` file.

Without a key the app uses a rough rule-based extractor, which is fine for trying the flow but will miss things. For reliable attribution, use speaker-labelled turns (`You:`, `ChatGPT:`, `Claude:`) or import the original `conversations.json`. If a pasted/uploaded document has no speaker labels, ChatCarry keeps it as unknown and does not guess which statements were the user's requirements or the assistant's suggestions. A DOCX transcript is not the same as the original Claude export; DOCX upload is not supported.

Optional settings (environment variables or `.env`): `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.8-flash`), `GEMINI_THINKING` (how hard Gemini thinks: `minimal`, `low`, `medium` or `high`; default `low`, which is faster; leave it empty to turn the setting off), `PORT`, `CHATCARRY_DATA` (folder for saved projects, default `projects` next to the app).

### Getting your chat out of ChatGPT or Claude

Request a data export in the app's settings, wait for the email, download the zip and **unzip it**. Then on the first page click **Import ChatGPT or Claude export (.json)**, choose `conversations.json`, and pick the conversation. Only these two formats are supported (not Gemini's Google Takeout).

## Tests

No key or internet needed:

```bash
python -m unittest discover -s tests -v      # backend, 104 tests
node --test tests/test_importers.js          # export importer, 9 tests (needs Node 18+)
```

## Troubleshooting

- **"Gemini did not answer within 120 seconds"**: Gemini was busy or slow. The page shows how many seconds a step has taken, and you can press Cancel at any time. Try again in a minute, or type a faster model (one with `flash-lite` in its name) into the Model box. Very long chats are cut into parts of about 10,000 characters, so each wait is for one part. The error names the part and the model. In Settings, press **Test Gemini speed**: if even that tiny question is slow, the problem is the model or the network, not your chat. A `flash-lite` model is usually the fastest.
- **"Gemini rejected the request (HTTP 400, 401 or 403)"**: the key or the model name is wrong. Check both.
- **The page looks old (no Import button, no My projects list)**: an older copy of the app is still running and holds port 5000. Close its black command window (or press Ctrl+C in it), then start the new one.
- **"Address already in use"**: same cause. Close the other ChatCarry window, or set `PORT=5001`.
- **Nothing is extracted**: check whether the text has speaker labels such as `You:` / `ChatGPT:`. If the source is an unlabelled document, ChatCarry will not guess who said what. Import the original `conversations.json` or add accurate speaker labels before extracting.

## How it works

```
chat (pasted, uploaded or imported)
   -> remove secrets and personal data
   -> split into numbered messages (You:/ChatGPT:/Claude:/... markers)
   -> take code blocks out (the app keeps them word for word)
   -> group the rest into chunks of about 10,000 characters
   -> per chunk: Gemini returns JSON with message-number citations
      (or the offline extractor, if there is no key)
      each step is streamed to the page, which drives the progress bar
   -> merge chunks: dedupe, newest code wins, drop invented citations
   -> analyse decisions: replaced vs conflicting
   -> you review and edit
   -> Markdown handoff, and the project saved to the projects folder
```

Files:

- `app.py`: all backend logic and the web routes
- `templates/index.html`: the whole interface, no build step
- `static/importers.js`: the ChatGPT / Claude export reader (runs in the browser)
- `start_chatcarry.bat`, `.env.example`: Windows launcher and settings template
- `sample_chat.txt`, `tests/`
- `projects/`: created when you first save a project

## Limits, stated honestly

- **The Gemini call has only been tried by hand on one project, and the v2.2 speed changes (code kept locally, smaller parts) have not been run against the live API.** The automated tests use fakes for the Gemini API. The app asks Gemini to think less (`GEMINI_THINKING`) to be faster; if Gemini rejects that setting the app quietly retries without it, but I could not check it against the live API. If a call fails, the app shows what Gemini said.
- **LLM extraction can be wrong.** That is why every item carries a source link and the review step exists. Measure it on your own chats: take 3 or 4 long ones and count how many extracted decisions are correct.
- **The export importer was tested on hand-made files in the documented ChatGPT and Claude formats, not on real exports.** Both companies change their export formats from time to time. If a real export fails or looks wrong, the parser is a small file (`static/importers.js`) that is easy to adjust.
- **The Windows launcher was written carefully but only checked on Linux.** If it fails, run the two commands under "Any system" instead.
- **The offline extractor is rough.** It looks for phrases like "let's use" and "next we need". It is a fallback, not a substitute for the model.
- **Redaction is pattern matching.** It catches common key formats, emails, phone numbers and `password = ...` lines. It cannot catch every secret, so skim your chat first.
- **Saved projects are plain JSON files, not encrypted.** They hold the extracted items and short excerpts of the cited messages (after redaction). Treat the `projects` folder like the chats themselves.
- **The conflict detector knows a fixed list of technologies** (databases, web frameworks, protocols, mobile frameworks) plus the topic and choice the model supplies. Contradictions about anything else are only caught when the model fills in those fields.
- **Pasted text needs speaker markers** such as `You:` and `ChatGPT:`. Without them the app splits on paragraphs and cannot tell who said what. Imported exports always have them.
- **Size limits:** 500,000 characters per chat (about 40 chunks) and 300 MB for an export file. Split bigger chats and merge the results through a saved project.
- **Cancel stops the page waiting.** The server finishes the part it is reading, then stops.
- The app is meant to run on your own computer. It listens on 127.0.0.1 only, refuses requests addressed to any other host name, and has no login. Do not expose it to the internet.

## Ideas for next steps

- A 1-page and a 5-page handoff for AIs with small context windows
- "What changed" view when a new chat is merged
- Task board view (to do / in progress / blocked / done)
- Domain templates (IoT and healthcare, research, hackathon)
- Browser extension to read the chat straight from the page
- Local model option, so nothing leaves the computer
- A self-check: a second AI reads only the handoff and answers questions about the project

## Compared with existing tools

Tools such as ContextBridge, SessionBridge, Cross AI and MemoryPlugin already move chats or keep a shared memory across AIs. ChatCarry focuses on the part they mostly skip: a reviewable, source-linked project state that stays correct over several sessions.

### Project-scoped extraction (v6.3)

When a pasted conversation discusses multiple projects, ChatCarry checks project headings and nearby sentence context before assigning technologies and project facts. Items clearly belonging to a related project are not copied into the current project's technology stack, decisions, completed work, bugs, requirements, or next tasks. Ambiguous ownership is treated conservatively. Testing and validation actions are kept under Next Tasks rather than product Requirements. Explicit successful test-run summaries can be recorded as tested completion evidence, without implying that the whole application is production-ready.

### Evidence-aware task consolidation (v6.4)

When the source repeats the same extraction-reliability work in different words, ChatCarry consolidates it into one focused task for technology classification, deduplication, completion evidence, or source references. It carries forward the source references from every merged item. Broad reliability umbrella tasks are split into the subtopics they explicitly mention; unrelated tasks remain separate.


### Strict technology ownership evidence (v6.5)

A model-suggested technology is excluded from the active project's stack if it cannot be traced to the pasted source. Related-project headings such as `IntentIQ project` or `IntentIQ application` are recognized so technologies listed beneath them remain with that project. This is intentionally conservative: a technology with unclear ownership is not silently assigned to ChatCarry.


### Escaped handoff heading recognition (v6.6)

Copied handoff content can contain escaped Markdown headings such as `\# PROJECT NAME` and `\## IntentIQ`. The extraction ownership checks now recognize these as structural headings. This prevents technologies listed in a related project's section from being treated as unscoped mentions and imported into the active project's Tech Stack.


### Task consolidation and provenance preservation (v6.7)

When one extracted task explicitly names multiple reliability topics, ChatCarry creates the corresponding focused tasks and preserves their shared source reference. A generic extraction-quality goal remains a single parent task unless it explicitly names subtopics. Merging duplicate tasks no longer truncates supporting source references. The v6.6 project-ownership and technology-isolation logic is retained unchanged.
