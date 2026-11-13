## v6.7 — Task consolidation and provenance preservation

- Consolidate one extracted task that explicitly names several reliability topics into the corresponding focused tasks, rather than leaving a broad multi-topic duplicate in Next Tasks.
- Keep a generic “improve extraction quality” goal as one parent-level task when it does not specify subtopics; do not falsely attach its source evidence to individual technical subtasks.
- Preserve all source references when exact or near-duplicate tasks are merged; removed the previous eight-reference cap and the twelve-reference cap in topic consolidation.
- Added regression tests based on the user's latest ChatCarry handoff task list, including checks that unrelated tasks are not merged and broad-task evidence is not misattributed.
- Existing project-ownership and technology-isolation logic from v6.6 is unchanged.

## v6.6 — Escaped Markdown project-scope recognition

- Project-heading detection now recognizes escaped Markdown heading markers (for example, `\# PROJECT NAME` and `\## IntentIQ`) commonly present in copied/exported handoff text.
- Technology-stack section detection recognizes the same escaped heading form, so current-project stack evidence is retained while related-project stack evidence remains isolated.
- Added regression tests using an escaped ChatCarry handoff containing Python, Flask, and Gemini API under ChatCarry and TensorFlow Lite, OpenCV, Whisper, WorkManager, and Lottie under IntentIQ.
- This fixes a concrete parsing gap in the extraction/ownership stage; it does not change the Review or Handoff workflow.

## v6.5 — Strict technology ownership evidence

- A technology proposed by the model is no longer accepted into the active project's stack when its name cannot be found anywhere in the supplied source. Missing evidence is treated as unknown ownership, not implicit ownership.
- Project-heading recognition now handles common suffixes such as `IntentIQ project`, `IntentIQ application`, `IntentIQ repo`, and `IntentIQ system`, preventing technologies under those related-project sections from leaking into the active project.
- Added regression tests for unmentioned model-generated technologies, project-heading suffixes, and the previously supported cross-project and evidence-consolidation cases.

# ChatCarry changes

## Evidence and source reliability update

- Added separate technology statuses for owner-approved, proposed, considered, in-use, implemented-but-untested, and implemented-and-verified technologies.
- Added completion evidence statuses so implemented work is not automatically described as tested.
- Tightened offline technology classification: questions are treated as considered; assistant-only recommendations do not become owner-adopted technologies.
- Made deduplication more conservative. Similar wording alone no longer merges facts; exact repeats, meaningful-token rearrangements, and clear repeated phrases with extra context can merge.
- Source references now retain whether they point to a speaker-labelled turn or an unlabelled source block. Handoffs explain when original turn boundaries cannot be reconstructed.
- Long-input rejection explicitly states that nothing was processed and all content was skipped when a hard limit is exceeded.
- Updated the README and added regression tests for these behaviours, plus JSON state roundtrip and Markdown handoff output.


## v6.1 — entity classification

- Added a separate Related projects / test context section so named applications and project examples are not misrepresented as technologies.
- Tightened the Gemini extraction instructions to classify entity type before assigning technology status and to omit ambiguous entities rather than guessing.
- Added conservative offline detection for explicitly named projects used in test/demo/example contexts.
- Added regression tests for project-versus-technology separation and handoff export.

## v6.2 — Cross-chunk entity and status corrections

- Reconcile entity roles after all extraction chunks are merged, so a named project cannot remain in Tech Stack merely because Gemini classified it differently in another chunk.
- Exclude a former project name from Related Projects when a naming decision explicitly replaces it with the current project name.
- Correct technologies listed as part of the current stack or used to build the app to `in_use_unverified`, rather than implying the technology itself was implemented.
- Strengthen the Gemini instructions for former project names and technology-status semantics.

## v6.3 — Project-scoped extraction reliability

- Added project-scope isolation across chunks: technologies, decisions, completed work, bugs, requirements, pending tasks, and do-not-change items tied to a related project are excluded from the main project's state unless explicitly connected to it.
- Added source-heading and sentence-context checks so technologies mentioned under a related project's section (for example, IntentIQ) do not leak into the current project's tech stack.
- Kept stack-status correction scoped to the main project's source section.
- Moved validation/testing actions (such as testing the current project's handoff or export/import) from Requirements to Next Tasks.
- Added conservative deduplication for umbrella extraction-reliability tasks while preserving uncovered subtopics such as evidence-based completion.
- Captured explicit successful unittest/pytest summaries as tested completion evidence with source references; this records the test suite result, not a claim that the entire application is production-ready.
- Expanded Gemini instructions to enforce project ownership and avoid transferring related-project decisions, technologies, bugs, or completed work.

## v6.4 — Evidence-aware task consolidation

- Consolidate repeated technology-classification, deduplication, completion-evidence, and source-reference tasks into one focused task per topic.
- Split broad extraction-reliability umbrella tasks into their explicitly named subtopics instead of keeping a redundant umbrella entry.
- Preserve and combine all contributing source references when task variants are consolidated.
- Keep unrelated validation tasks separate and move validation actions mistakenly extracted as requirements into Next Tasks.
- Added regression tests for overlapping task variants, umbrella-task splitting, source-reference preservation, and requirement-to-task correction.
