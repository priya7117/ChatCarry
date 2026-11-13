"""Tests for ChatCarry. Run from the project folder with:  python -m unittest discover -s tests -v

The Gemini API is never called: call_gemini / urlopen are replaced with fakes.
"""
import io
import json
import os
import re
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as ca  # noqa: E402

FAKE_KEY = "AIzaSyTESTKEY0123456789abcdefghijklmnop"


def chat(*turns):
    return "\n\n".join(f"{who}: {text}" for who, text in turns)


def state_with(decisions=(), tech=()):
    st = ca.empty_state()
    st["decisions"] = [{"text": t, "src": [], "reason": "", "topic": "", "choice": ""} for t in decisions]
    st["tech_stack"] = [{"text": t, "src": []} for t in tech]
    return st


class RedactionTests(unittest.TestCase):

    def test_tech_stack_status_is_preserved_and_validated(self):
        allowed = ("approved", "implemented_unverified", "implemented_verified", "in_use_unverified", "proposed", "considered", "rejected", "unknown")
        for status in allowed:
            item = ca.clean_item({"text": "Room", "status": status, "src": [1]}, "tech_stack", lambda x: ["S1#1"])
            self.assertEqual(status, item["status"])
        legacy_active = ca.clean_item({"text": "Kotlin", "status": "active"}, "tech_stack", lambda x: [])
        self.assertEqual("in_use_unverified", legacy_active["status"])
        invalid = ca.clean_item({"text": "Lottie", "status": "made-up"}, "tech_stack", lambda x: [])
        self.assertEqual("unknown", invalid["status"])

    def test_unknown_tech_status_does_not_overwrite_known_status(self):
        items = [
            {"text": "Kotlin", "status": "in_use_unverified", "src": ["S1#1"]},
            {"text": "Kotlin", "status": "unknown", "src": ["S1#2"]},
        ]
        result = ca.dedupe(items)
        self.assertEqual(1, len(result))
        self.assertEqual("in_use_unverified", result[0]["status"])

    def test_dedupe_ignores_word_order_for_close_duplicates(self):
        items = [
            {"text": "Change project name from Focus Warrior to IntentIQ", "src": ["S1#1"]},
            {"text": "Change the name of the project from Focus Warrior to IntentIQ", "src": ["S1#2"]},
        ]
        result = ca.dedupe(items)
        self.assertEqual(1, len(result))
        self.assertEqual({"S1#1", "S1#2"}, set(result[0]["src"]))

    def test_handoff_includes_technology_status(self):
        st = ca.empty_state()
        st["name"] = "Demo"
        st["tech_stack"] = [
            {"text": "Kotlin", "status": "in_use_unverified", "src": []},
            {"text": "TensorFlow Lite", "status": "proposed", "src": []},
        ]
        md = ca.build_handoff(st)
        self.assertIn("[In use; not yet verified] Kotlin", md)
        self.assertIn("[Proposed] TensorFlow Lite", md)
    def test_handoff_separates_related_projects_from_tech_stack(self):
        st = ca.empty_state()
        st["name"] = "ChatCarry"
        st["tech_stack"] = [{"text": "Flask", "status": "in_use_unverified", "src": []}]
        st["related_projects"] = [{"text": "IntentIQ", "src": []}]
        md = ca.build_handoff(st)
        self.assertIn("## TECH STACK", md)
        self.assertIn("## RELATED PROJECTS / TEST CONTEXT", md)
        self.assertNotIn("[Status unknown; verify] IntentIQ", md)

    def test_secrets_and_personal_data_removed(self):
        text = ("mail me at priya.s@example.com or call +91 98765 43210 / 9876543210. "
                f"key {FAKE_KEY} and password = hunter2x. Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n"
                "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----")
        out, counts = ca.redact(text)
        for secret in ["priya.s@example.com", "98765 43210", "9876543210", FAKE_KEY, "hunter2x",
                       "abcdefghijklmnopqrstuvwxyz123456", "MIIBOgIBAAJBAK"]:
            self.assertNotIn(secret, out)
        self.assertEqual({"email", "phone", "api_key", "password_or_secret", "bearer_token", "private_key"}, set(counts))

    def test_code_is_not_mangled(self):
        code = ('token = request.args.get("t")\npassword = input("Password: ")\n'
                "host = '192.168.100.200'\nPython 3.11.4 on port 1883\nusers = os.environ['API_KEY']")
        out, counts = ca.redact(code)
        self.assertEqual(code, out)
        self.assertEqual({}, counts)


class SplitTests(unittest.TestCase):
    def test_roles_and_numbering(self):
        msgs = ca.split_messages(chat(("You", "hello there"), ("ChatGPT", "hi"), ("You", "next")))
        self.assertEqual([(1, "user"), (2, "assistant"), (3, "user")], [(m["n"], m["role"]) for m in msgs])

    def test_you_said_format_and_bold_markers(self):
        msgs = ca.split_messages("**You said:** first\n\n**ChatGPT said:**\nsecond line\nthird")
        self.assertEqual(2, len(msgs))
        self.assertEqual("second line\nthird", msgs[1]["text"])

    def test_marker_inside_code_fence_does_not_split(self):
        text = "You: look\n\nChatGPT: here\n```\nUser: not a speaker\nAI: still code\n```\nYou: thanks"
        msgs = ca.split_messages(text)
        self.assertEqual(3, len(msgs))
        self.assertIn("User: not a speaker", msgs[1]["text"])

    def test_fallback_without_markers(self):
        msgs = ca.split_messages("just some notes\n\nmore notes here\n\nand a third paragraph")
        self.assertTrue(msgs)
        self.assertEqual("unknown", msgs[0]["role"])

    def test_unlabelled_claude_document_is_one_unknown_message(self):
        fixture = Path(ca.__file__).parent / "tests" / "fixtures" / "claude_unlabelled.txt"
        text = fixture.read_text(encoding="utf-8")
        msgs = ca.split_messages(text)
        self.assertEqual(1, len(msgs))
        self.assertEqual("unknown", msgs[0]["role"])
        self.assertEqual(text.strip(), msgs[0]["text"])

    def test_unlabelled_text_can_produce_candidate_project_state(self):
        msgs = ca.split_messages(
            "Claude\\n\\nI built a Flask app and recommend PostgreSQL.\\n\\n"
            "how to run this\\n\\nYou should add login and tests next."
        )
        self.assertEqual(["unknown"], [m["role"] for m in msgs])
        result = ca.heuristic_extract(msgs)
        self.assertTrue(result["tech_stack"] or result["pending"] or result["completed"])

    def test_related_project_is_not_added_to_technology_stack(self):
        msgs = ca.split_messages(
            "I tested ChatCarry using IntentIQ.\n\n"
            "Python and Flask are currently in use for ChatCarry."
        )
        result = ca.heuristic_extract(msgs)
        self.assertIn("IntentIQ", [item["text"] for item in result["related_projects"]])
        self.assertNotIn("IntentIQ", [item["text"] for item in result["tech_stack"]])
        self.assertIn("Python", [item["text"] for item in result["tech_stack"]])
        self.assertIn("Flask", [item["text"] for item in result["tech_stack"]])

    def test_gemini_prompt_separates_projects_from_technologies(self):
        self.assertIn("A named application, product, project", ca.SYSTEM_PROMPT)
        self.assertIn('"related_projects"', ca.SYSTEM_PROMPT)

    def test_related_project_classification_wins_if_model_duplicates_entity(self):
        raw = {
            "tech_stack": [{"text": "Flask", "status": "in_use_unverified", "src": [1]},
                           {"text": "IntentIQ", "status": "unknown", "src": [1]}],
            "related_projects": [{"text": "IntentIQ", "src": [1]}],
        }
        result = ca.normalize_partial(raw, 1, {1})
        self.assertEqual(["Flask"], [item["text"] for item in result["tech_stack"]])
        self.assertEqual(["IntentIQ"], [item["text"] for item in result["related_projects"]])

    def test_cross_chunk_related_project_removes_technology_duplicate(self):
        state = ca.empty_state()
        state["tech_stack"] = [{"text": "IntentIQ", "status": "considered", "src": ["S1#1"]},
                               {"text": "Python", "status": "implemented_unverified", "src": ["S1#1"]}]
        state["related_projects"] = [{"text": "IntentIQ", "src": ["S1#2"]}]
        result = ca.finalize(state)
        self.assertEqual(["Python"], [item["text"] for item in result["tech_stack"]])

    def test_former_project_name_is_not_a_related_project(self):
        state = ca.empty_state()
        state["name"] = "ChatCarry"
        state["related_projects"] = [{"text": "ContextBridge", "src": ["S1#1"]},
                                     {"text": "IntentIQ", "src": ["S1#2"]}]
        state["decisions"] = [{"text": "Use the name ChatCarry instead of ContextBridge", "reason": "simpler name", "topic": "", "choice": "", "src": ["S1#1"]}]
        result = ca.finalize(state)
        self.assertEqual(["IntentIQ"], [item["text"] for item in result["related_projects"]])

    def test_current_stack_context_corrects_implemented_status_to_in_use(self):
        raw = {"tech_stack": [{"text": "Python", "status": "implemented_unverified", "src": [1]},
                              {"text": "Flask", "status": "implemented_unverified", "src": [1]}]}
        msgs = [{"n": 1, "role": "unknown", "text": "Tech stack: Python, Flask, and Gemini API."}]
        result = ca.normalize_partial(raw, 1, {1}, msgs)
        statuses = {item["text"]: item["status"] for item in result["tech_stack"]}
        self.assertEqual("in_use_unverified", statuses["Python"])
        self.assertEqual("in_use_unverified", statuses["Flask"])

    def test_project_scope_excludes_related_project_technologies(self):
        source = """# PROJECT STATE — ChatCarry
## TECH STACK
Python, Flask, Gemini API
## RELATED PROJECTS / TEST CONTEXT
IntentIQ
## IntentIQ
### TECH STACK
TensorFlow Lite, OpenCV, Whisper, WorkManager, Lottie
"""
        raw = {"name": "ChatCarry", "tech_stack": [
            {"text": "Python", "status": "in_use_unverified", "src": [1]},
            {"text": "Flask", "status": "in_use_unverified", "src": [1]},
            {"text": "Gemini API", "status": "in_use_unverified", "src": [1]},
            {"text": "TensorFlow Lite", "status": "proposed", "src": [1]},
            {"text": "OpenCV", "status": "proposed", "src": [1]},
            {"text": "Whisper", "status": "proposed", "src": [1]},
            {"text": "WorkManager", "status": "proposed", "src": [1]},
            {"text": "Lottie", "status": "proposed", "src": [1]},
        ], "related_projects": [{"text": "IntentIQ", "src": [1]}]}
        msgs = [{"n": 1, "role": "unknown", "text": source}]
        result = ca.normalize_partial(raw, 1, {1}, msgs)
        self.assertEqual(["Python", "Flask", "Gemini API"], [x["text"] for x in result["tech_stack"]])

    def test_escaped_markdown_handoff_headings_isolate_related_project_technologies(self):
        # Handoff/export text can contain escaped Markdown markers (\\#).
        # These are still structural headings for project-ownership checks.
        source = r"""\# PROJECT NAME — ChatCarry
\## GOAL
Build a dependable project handoff tool.
\## TECH STACK
- Python
- Flask
- Gemini API
\## RELATED PROJECTS / TEST CONTEXT
- IntentIQ
\## IntentIQ
\### TECH STACK
- TensorFlow Lite
- OpenCV
- Whisper
- WorkManager
- Lottie
"""
        raw = {"name": "ChatCarry", "tech_stack": [
            {"text": "Python", "status": "in_use_unverified", "src": [1]},
            {"text": "Flask", "status": "in_use_unverified", "src": [1]},
            {"text": "Gemini API", "status": "in_use_unverified", "src": [1]},
            {"text": "TensorFlow Lite", "status": "proposed", "src": [1]},
            {"text": "OpenCV", "status": "proposed", "src": [1]},
            {"text": "Whisper", "status": "proposed", "src": [1]},
            {"text": "WorkManager", "status": "proposed", "src": [1]},
            {"text": "Lottie", "status": "proposed", "src": [1]},
        ], "related_projects": [{"text": "IntentIQ", "src": [1]}]}
        messages = [{"n": 1, "role": "unknown", "text": source}]
        result = ca.normalize_partial(raw, 1, {1}, messages)
        self.assertEqual(["Python", "Flask", "Gemini API"],
                         [item["text"] for item in result["tech_stack"]])
        self.assertEqual("in_use_unverified", next(
            item["status"] for item in result["tech_stack"] if item["text"] == "Python"))

    def test_escaped_current_tech_stack_heading_is_recognized(self):
        source = r"""\# PROJECT NAME — ChatCarry
\## TECH STACK
Python, Flask, Gemini API
\## RELATED PROJECTS / TEST CONTEXT
IntentIQ
"""
        main_scope = "\n".join(ca._main_project_source_texts(
            [{"n": 1, "role": "unknown", "text": source}], "ChatCarry", ["IntentIQ"]))
        self.assertTrue(ca._in_current_stack_section(main_scope, "Python"))
        self.assertFalse(ca._in_current_stack_section(main_scope, "Whisper"))

    def test_project_scope_uses_sentence_context_without_markdown_headings(self):
        source = "IntentIQ uses Whisper for transcription. ChatCarry uses Python and Flask."
        self.assertFalse(ca._tech_is_in_main_project("Whisper", [{"text": source}], "ChatCarry", ["IntentIQ"]))
        self.assertTrue(ca._tech_is_in_main_project("Python", [{"text": source}], "ChatCarry", ["IntentIQ"]))

    def test_unmentioned_model_technology_is_not_added_to_project_stack(self):
        source = "ChatCarry uses Python and Flask. IntentIQ is a related project."
        self.assertFalse(ca._tech_is_in_main_project("TensorFlow Lite", [{"text": source}], "ChatCarry", ["IntentIQ"]))

    def test_related_project_heading_with_project_suffix_is_recognized(self):
        source = """# PROJECT STATE — ChatCarry
## ChatCarry project
### TECH STACK
Python, Flask
## IntentIQ project
### TECH STACK
TensorFlow Lite, OpenCV, Whisper
"""
        self.assertFalse(ca._tech_is_in_main_project("Whisper", [{"text": source}], "ChatCarry", ["IntentIQ"]))
        self.assertTrue(ca._tech_is_in_main_project("Python", [{"text": source}], "ChatCarry", ["IntentIQ"]))

    def test_normalization_drops_unmentioned_model_technology(self):
        source = "# PROJECT STATE — ChatCarry\n## TECH STACK\nPython, Flask\n## RELATED PROJECTS / TEST CONTEXT\nIntentIQ"
        raw = {"name": "ChatCarry", "tech_stack": [
            {"text": "Python", "status": "in_use_unverified", "src": [1]},
            {"text": "Flask", "status": "in_use_unverified", "src": [1]},
            {"text": "TensorFlow Lite", "status": "proposed", "src": [1]},
        ], "related_projects": [{"text": "IntentIQ", "src": [1]}]}
        result = ca.normalize_partial(raw, 1, {1}, [{"n": 1, "role": "unknown", "text": source}])
        self.assertEqual(["Python", "Flask"], [x["text"] for x in result["tech_stack"]])

    def test_related_project_test_summary_is_not_main_project_completion(self):
        state = ca.empty_state()
        state["name"] = "ChatCarry"
        state["related_projects"] = [{"text": "IntentIQ", "src": ["S2#1"]}]
        source = """# PROJECT STATE — ChatCarry
## IntentIQ
Ran 24 tests in 0.12s

OK
"""
        ca._extract_test_run_evidence(state, [{"n": 1, "text": source}], session=2)
        self.assertEqual([], state["completed"])

    def test_related_project_completed_work_and_decisions_do_not_leak(self):
        source = """# PROJECT STATE — ChatCarry
## GOAL
Build ChatCarry.
## IntentIQ
### COMPLETED
Implemented the image classifier for IntentIQ.
### DECISIONS
Use TensorFlow Lite for IntentIQ.
## ChatCarry
### COMPLETED
Implemented project-scoped extraction for ChatCarry.
"""
        state = ca.empty_state()
        state["name"] = "ChatCarry"
        state["related_projects"] = [{"text": "IntentIQ", "src": ["S1#1"]}]
        state["completed"] = [
            {"text": "Implemented the image classifier for IntentIQ.", "status": "implemented_unverified", "src": ["S1#1"]},
            {"text": "Implemented project-scoped extraction for ChatCarry.", "status": "implemented_unverified", "src": ["S1#1"]},
        ]
        state["decisions"] = [
            {"text": "Use TensorFlow Lite for IntentIQ.", "reason": "", "topic": "", "choice": "", "src": ["S1#1"]}
        ]
        result = ca.finalize(state, [{"n": 1, "role": "unknown", "text": source}])
        self.assertEqual(["Implemented project-scoped extraction for ChatCarry."],
                         [x["text"] for x in result["completed"]])
        self.assertEqual([], result["decisions"])

    def test_validation_requirement_is_moved_to_pending(self):
        state = ca.empty_state()
        state["requirements"] = [{"text": "We must test ChatCarry's own handoff using a ChatCarry conversation.", "src": ["S1#1"]}]
        ca.finalize(state)
        self.assertEqual([], state["requirements"])
        self.assertTrue(any("test ChatCarry" in x["text"] for x in state["pending"]))

    def test_successful_unittest_summary_becomes_tested_completion(self):
        state = ca.empty_state()
        state["sessions"] = 1
        messages = [{"n": 3, "role": "user", "text": "Ran 91 tests in 0.418s\n\nOK"}]
        ca._extract_test_run_evidence(state, messages, session=2)
        self.assertEqual(1, len(state["completed"]))
        self.assertEqual("tested_verified", state["completed"][0]["status"])
        self.assertIn("91 tests passed", state["completed"][0]["text"])
        self.assertEqual(["S2#3"], state["completed"][0]["src"])

    def test_redundant_extraction_reliability_task_preserves_missing_child(self):
        state = ca.empty_state()
        state["pending"] = [
            {"text": "Improve technology status classification.", "src": ["S1#1"]},
            {"text": "Improve deduplication of extracted items.", "src": ["S1#1"]},
            {"text": "Improve source references.", "src": ["S1#1"]},
            {"text": "Improve extraction reliability, including technology status classification, duplicate items, evidence-based completion, and honest source references.", "src": ["S1#1"]},
        ]
        ca.finalize(state)
        texts = [x["text"] for x in state["pending"]]
        self.assertFalse(any("Improve extraction reliability" in x for x in texts))
        self.assertTrue(any("evidence-based completion" in x for x in texts))
        self.assertEqual(4, len(texts))

    def test_unknown_segments_are_isolated_from_labelled_turns(self):
        msgs = ca.split_messages("Unlabelled preamble.\n\nYou: Use SQLite.\n\nChatGPT: okay.")
        chunks = ca.make_chunks(msgs)
        chunk_roles = []
        for chunk, nums in chunks:
            chunk_roles.append({m["role"] for m in msgs if m["n"] in nums})
        self.assertEqual([{"unknown"}, {"user", "assistant"}], chunk_roles)

    def test_speakerless_gemini_uses_normal_project_extraction(self):
        fixture = Path(ca.__file__).parent / "tests" / "fixtures" / "claude_unlabelled.txt"
        text = fixture.read_text(encoding="utf-8")

        def fake_extract(api_key, model, user_text, chunk_index,
                         total_chunks, message_nums, system=None):
            self.assertIn("project history", system.lower())
            return {"goal": "Carry project state between AI chats", "tech_stack": [
                {"text": "Python", "src": [message_nums[0]]}
            ]}

        with mock.patch.object(ca, "llm_extract_chunk", side_effect=fake_extract) as llm:
            events = list(ca.extract_events({"text": text, "mode": "gemini", "api_key": FAKE_KEY}))
        llm.assert_called()
        result = next(e for e in events if e["type"] == "result")
        self.assertEqual("gemini", result["report"]["mode"])
        self.assertTrue(result["state"]["goal"])
        self.assertTrue(any("no reliable speaker labels" in w.lower()
                            for w in result["report"]["warnings"]))
        self.assertEqual(1, result["report"]["source_blocks"])
        self.assertEqual(0, result["report"]["labelled_messages"])

    def test_chunking_keeps_numbers_and_splits_big_messages(self):
        msgs = ca.split_messages(chat(("You", "a" * 40000), ("ChatGPT", "short reply")))
        chunks = ca.make_chunks(msgs)
        self.assertGreaterEqual(len(chunks), 3)
        for text, nums in chunks:
            self.assertLessEqual(len(text), ca.CHUNK_CHARS + 200)
        self.assertEqual({1, 2}, {n for _, nums in chunks for n in nums})


class MergeTests(unittest.TestCase):
    def test_pending_reliability_tasks_are_consolidated_with_all_sources(self):
        state = ca.empty_state()
        state["pending"] = [
            {"text": "Improve technology classification to avoid falsely claiming discussions are implemented technologies.", "src": ["S1#2"]},
            {"text": "Improve technology classification and status assignment.", "src": ["S1#7"]},
            {"text": "Improve deduplication of extracted items.", "src": ["S1#3"]},
            {"text": "Improve conservative deduplication without merging similar but distinct statements.", "src": ["S1#8"]},
            {"text": "Improve source references when exact message-level references cannot be reconstructed.", "src": ["S1#4"]},
            {"text": "Improve honest source references for unlabeled blocks.", "src": ["S1#9"]},
            {"text": "Manually test the browser workflow.", "src": ["S1#10"]},
        ]
        out = ca._reconcile_pending_tasks(state)
        texts = [x["text"] for x in out["pending"]]
        self.assertEqual(1, sum("technology classification" in x.lower() for x in texts))
        self.assertEqual(1, sum("deduplication" in x.lower() for x in texts))
        self.assertEqual(1, sum("source references" in x.lower() for x in texts))
        tech = next(x for x in out["pending"] if "technology classification" in x["text"].lower())
        self.assertEqual(["S1#2", "S1#7"], tech["src"])
        self.assertTrue(any(x.get("text") == "Manually test the browser workflow." for x in out["pending"]))

    def test_actual_chatcarry_handoff_tasks_are_consolidated_without_false_topic_evidence(self):
        state = ca.empty_state()
        state["pending"] = [
            {"text": "Improve extraction quality of the current version", "src": ["S1#1"]},
            {"text": "Improve technology classification, duplicate removal, and source references", "src": ["S1#2"]},
            {"text": "Improve technology classification and status assignment, ensuring discussed or suggested technologies are not incorrectly marked as implemented.", "src": ["S1#3"]},
            {"text": "Improve conservative deduplication so repeated tasks are merged without combining distinct but similar statements.", "src": ["S1#4"]},
            {"text": "Improve evidence-based completion and test verification; distinguish planned, implemented, and successfully tested work.", "src": ["S1#5"]},
            {"text": "Improve honest source references when original turns or exact supporting passages cannot be reconstructed.", "src": ["S1#6"]},
        ]
        out = ca._reconcile_pending_tasks(state)
        texts = [item["text"].lower() for item in out["pending"]]
        self.assertEqual(1, sum("overall extraction quality" in text for text in texts))
        self.assertEqual(1, sum("technology classification" in text for text in texts))
        self.assertEqual(1, sum("deduplication" in text for text in texts))
        self.assertEqual(1, sum("evidence-based completion" in text for text in texts))
        self.assertEqual(1, sum("source references" in text for text in texts))
        quality = next(item for item in out["pending"] if "overall extraction quality" in item["text"].lower())
        self.assertEqual(["S1#1"], quality["src"])
        tech = next(item for item in out["pending"] if "technology classification" in item["text"].lower())
        self.assertEqual(["S1#2", "S1#3"], tech["src"])

    def test_dedupe_does_not_truncate_supporting_source_references(self):
        items = [
            {"text": "Improve conservative task deduplication across chunks", "src": [f"S1#{n}"]}
            for n in range(1, 13)
        ]
        result = ca.dedupe(items)
        self.assertEqual(1, len(result))
        self.assertEqual(12, len(result[0]["src"]))

    def test_reliability_umbrella_is_split_into_focused_topics(self):
        state = ca.empty_state()
        state["pending"] = [
            {"text": "Improve extraction reliability, specifically technology status classification, duplicate items, evidence-based completion, and honest source references.", "src": ["S1#5"]}
        ]
        out = ca._reconcile_pending_tasks(state)
        texts = [x["text"].lower() for x in out["pending"]]
        for topic in ("technology classification", "deduplication", "completion", "source references"):
            self.assertEqual(1, sum(topic in text for text in texts), topic)
        self.assertFalse(any("extraction reliability" in text for text in texts))

    def test_validation_task_moves_from_requirement_to_pending(self):
        state = ca.empty_state()
        state["requirements"] = [{"text": "Test ChatCarry's own handoff using a ChatCarry conversation.", "src": ["S1#6"]}]
        out = ca._reconcile_pending_tasks(state)
        self.assertEqual([], out["requirements"])
        self.assertTrue(any("Test ChatCarry's own handoff" in x["text"] for x in out["pending"]))

    def test_dedupe_merges_sources_and_keeps_latest_bug_status(self):
        items = [{"text": "Alert does not trigger above 120 bpm", "src": ["S1#2"], "status": "open"},
                 {"text": "Alert does not trigger above 120 bpm.", "src": ["S1#9"], "status": "fixed"}]
        out = ca.dedupe(items)
        self.assertEqual(1, len(out))
        self.assertEqual("fixed", out[0]["status"])
        self.assertEqual(["S1#2", "S1#9"], out[0]["src"])

    def test_near_duplicate_privacy_requirements_are_merged(self):
        items = [
            {"text": "Do not collect private messages, passwords, or personal content; ensure strict privacy.", "src": ["S1#1"]},
            {"text": "Do not collect private messages, passwords, or personal content.", "src": ["S1#1"]},
        ]
        out = ca.dedupe(items)
        self.assertEqual(1, len(out))
        self.assertEqual(["S1#1"], out[0]["src"])

    def test_similar_words_do_not_merge_distinct_requirements(self):
        items = [
            {"text": "Use Flask for the public API endpoint", "src": ["S1#1"]},
            {"text": "Do not use Flask for the public API endpoint", "src": ["S1#2"]},
        ]
        self.assertEqual(2, len(ca.dedupe(items)))

    def test_items_with_different_numbers_are_never_merged(self):
        items = [{"text": "The alert threshold is 120 bpm", "src": ["S1#2"]},
                 {"text": "The alert threshold is 130 bpm", "src": ["S1#9"]}]
        self.assertEqual(2, len(ca.dedupe(items)))

    def test_newest_code_version_wins(self):
        blocks = [{"file": "a.py", "language": "python", "code": "v2", "status": "ok", "src": ["S1#8"], "versions": 1},
                  {"file": "A.py", "language": "python", "code": "v1", "status": "ok", "src": ["S1#2"], "versions": 1},
                  {"file": "b.py", "language": "python", "code": "b", "status": "ok", "src": ["S1#4"], "versions": 1}]
        out = ca.track_code(blocks)
        self.assertEqual({"a.py": "v2", "b.py": "b"}, {c["file"]: c["code"] for c in out})
        self.assertEqual(2, next(c for c in out if c["file"] == "a.py")["versions"])

    def test_unlabelled_snippets_with_same_first_line_stay_separate(self):
        blocks = [
            {"file": "", "language": "python", "code": "import os\nvalue = 1\nprint(value)", "status": "unknown", "src": ["S1#1"], "versions": 1},
            {"file": "", "language": "python", "code": "import os\nname = 'Priya'\nprint(name)", "status": "unknown", "src": ["S1#2"], "versions": 1},
        ]
        self.assertEqual(2, len(ca.track_code(blocks)))


class DecisionAnalysisTests(unittest.TestCase):
    def test_silent_contradiction_is_a_conflict(self):
        a = ca.analyze_decisions(state_with(["Let's use Firebase for the database.", "Let's use MongoDB for the database."]))
        self.assertEqual(1, len(a["conflicts"]))
        self.assertEqual("MongoDB", a["conflicts"][0]["latest"])
        self.assertEqual({0, 1}, {o["decision_index"] for o in a["conflicts"][0]["options"]})
        self.assertEqual([], a["superseded"])

    def test_explicit_switch_replaces_instead_of_conflicting(self):
        st = state_with(["Let's use Firebase for the database.", "Let's use SQLite instead of Firebase."], tech=["Firebase", "SQLite"])
        a = ca.analyze_decisions(st)
        self.assertEqual([], a["conflicts"])
        self.assertEqual([0], a["superseded"])
        self.assertEqual([0], a["stale_tech"])

    def test_not_use_anymore_replaces(self):
        a = ca.analyze_decisions(state_with(["Let's use Firebase.", "We will not use Firebase anymore."]))
        self.assertEqual([0], a["superseded"])
        self.assertEqual([], a["conflicts"])

    def test_single_comparison_is_not_a_conflict(self):
        a = ca.analyze_decisions(state_with(["We chose MQTT over HTTP because of latency."]))
        self.assertEqual([], a["conflicts"])

    def test_rechoosing_after_rejection_is_live_again(self):
        a = ca.analyze_decisions(state_with(["Use Firebase.", "Use SQLite instead of Firebase.", "Actually let's use Firebase again."]))
        self.assertEqual([], a["superseded"][1:])
        self.assertEqual("Firebase", ca.decision_choices("Actually let's use Firebase again.")[0]["database"])

    def test_llm_topic_and_choice_used_when_keywords_miss(self):
        st = ca.empty_state()
        st["decisions"] = [{"text": "We will deploy to Heroku", "src": [], "topic": "hosting", "choice": "Heroku", "reason": ""},
                           {"text": "Deploy on Render", "src": [], "topic": "hosting", "choice": "Render", "reason": ""}]
        self.assertEqual("hosting", ca.analyze_decisions(st)["conflicts"][0]["topic"])


class HandoffTests(unittest.TestCase):
    def setUp(self):
        st = state_with(["Let's use Firebase.", "Let's use SQLite instead of Firebase because offline demo."], tech=["Firebase", "SQLite"])
        st["name"], st["goal"] = "Demo", "Build a dashboard"
        st["code"] = [{"file": "x.md", "language": "md", "code": "```py\nprint(1)\n```", "status": "failing", "src": [], "versions": 1}]
        self.state = st

    def test_replaced_decisions_and_stale_tech_are_separated(self):
        md = ca.build_handoff(self.state, "gemini")
        active = md.split("## DECISIONS (and why)")[1].split("##")[0]
        replaced = md.split("## REPLACED EARLIER DECISIONS (do not use)")[1].split("##")[0]
        self.assertNotIn("Let's use Firebase.", active)
        self.assertIn("Let's use Firebase.", replaced)
        tech = md.split("## TECH STACK")[1].split("##")[0]
        self.assertNotIn("Firebase", tech)
        self.assertIn("SQLite", tech)

    def test_code_with_backticks_gets_longer_fence(self):
        md = ca.build_handoff(self.state)
        self.assertIn("````md", md)
        self.assertIn("status: failing", md)

    def test_unresolved_conflict_is_called_out(self):
        md = ca.build_handoff(state_with(["Use React.", "Use Vue."]))
        self.assertIn("UNRESOLVED CONFLICTS", md)
        self.assertIn("Ask me which one is final", md)

    def test_targets_and_verified_flag(self):
        self.assertIn("<project_state>", ca.build_handoff(self.state, "claude"))
        self.assertNotIn("<project_state>", ca.build_handoff(self.state, "chatgpt"))
        self.assertIn("not confirmed", ca.build_handoff(self.state, verified=False))
        self.assertIn("user before handoff: yes", ca.build_handoff(self.state, verified=True))


def fake_gemini(items_by_chunk=None):
    """A call_gemini replacement that cites real message numbers plus one invented number."""
    def fake(api_key, model, user_text, system=None):
        nums = [int(n) for n in re.findall(r"\[#(\d+) ", user_text)]
        chunk_no = int(re.search(r"chunk (\d+) of", user_text).group(1))
        body = {
            "goal": "Build a monitoring dashboard" if chunk_no == 2 else "",
            "tech_stack": [{"text": "Flask", "src": [nums[0]]}],
            "decisions": [{"text": "Use SQLite for storage", "reason": "offline demo", "topic": "database",
                           "choice": "SQLite", "src": [nums[0]]},
                          {"text": "Invented decision with a made-up source", "src": [9999]}],
            "bugs": [{"text": f"Bug found in chunk {chunk_no}", "status": "open", "src": [nums[-1]]}],
            "code": [{"file": "evil.py", "language": "python", "code": "TAMPERED", "status": "ok", "src": [nums[-1]]}],
            "code_status": [{"id": int(i), "status": "ok", "file": "model.py"}
                            for i in re.findall(r"\[code block C(\d+)", user_text)],
        }
        return "```json\n" + json.dumps(body) + "\n```"
    return fake


class HeuristicExtractionTests(unittest.TestCase):
    def test_offline_tech_status_uses_explicit_context(self):
        msgs = ca.split_messages(chat(
            ("You", "We are using Python in the backend."),
            ("You", "We plan to use Flask for the API."),
            ("You", "We are considering FastAPI as an alternative."),
        ))
        result = ca.heuristic_extract(msgs)
        statuses = {item["text"]: item["status"] for item in result["tech_stack"]}
        self.assertEqual("in_use_unverified", statuses["Python"])
        self.assertEqual("proposed", statuses["Flask"])
        self.assertEqual("considered", statuses["FastAPI"])

    def test_offline_name_requires_explicit_project_naming(self):
        named = ca.heuristic_extract(ca.split_messages(chat(("You", "Project name: ChatCarry."))))
        unnamed = ca.heuristic_extract(ca.split_messages(chat(("You", "I am building a Python web app."))))
        self.assertEqual("ChatCarry", named["name"])
        self.assertEqual("", unnamed["name"])

    def test_technology_mentions_are_not_assumed_implemented(self):
        msgs = ca.split_messages(chat(
            ("You", "Should we use TensorFlow Lite?"),
            ("ChatGPT", "I recommend OpenCV for image processing."),
            ("You", "I am considering Whisper as another option."),
        ))
        statuses = {x["text"]: x["status"] for x in ca.heuristic_extract(msgs)["tech_stack"]}
        self.assertEqual("considered", statuses["TensorFlow Lite"])
        self.assertNotIn("TensorFlow", statuses)
        self.assertNotIn("OpenCV", statuses)  # assistant-only recommendation is not owner adoption
        self.assertEqual("considered", statuses["Whisper"])

    def test_technology_approval_implementation_and_testing_are_distinct(self):
        msgs = ca.split_messages(chat(
            ("You", "We decided to use Flask for the backend."),
            ("You", "I implemented SQLite storage, but have not tested it yet."),
            ("You", "I successfully tested Python and the tests passed."),
        ))
        statuses = {x["text"]: x["status"] for x in ca.heuristic_extract(msgs)["tech_stack"]}
        self.assertEqual("approved", statuses["Flask"])
        self.assertEqual("implemented_unverified", statuses["SQLite"])
        self.assertEqual("implemented_verified", statuses["Python"])

    def test_completion_status_distinguishes_implementation_from_successful_tests(self):
        msgs = ca.split_messages(chat(
            ("You", "I implemented the export feature, but I have not tested it."),
            ("You", "The JSON import tests passed successfully."),
            ("ChatGPT", "The UI redesign is complete."),
        ))
        completed = ca.heuristic_extract(msgs)["completed"]
        by_text = {item["text"]: item["status"] for item in completed}
        self.assertEqual("implemented_unverified", by_text["I implemented the export feature, but I have not tested it."])
        self.assertEqual("tested_verified", by_text["The JSON import tests passed successfully."])
        self.assertFalse(any("UI redesign" in text for text in by_text))

    def test_speakerless_source_reference_is_not_described_as_a_chat_turn(self):
        transcript = "Project notes\n\nWe are using Flask for the local server.\n\nThe next task is to test the export."
        events = list(ca.extract_events({"text": transcript, "mode": "offline"}))
        state = next(e["state"] for e in events if e["type"] == "result")
        source = state["sources"]["S1#1"]
        self.assertEqual("unlabelled_source_block", source["kind"])
        self.assertIn("not an original chat turn", source["label"])
        self.assertIn("not an exact quoted passage", source["excerpt_note"])

    def test_source_reference_metadata_survives_json_roundtrip(self):
        state = ca.empty_state()
        state["sources"]["S1#1"] = {"role": "unknown", "kind": "unlabelled_source_block",
            "label": "Unlabelled source block (not an original chat turn)", "text": "opening excerpt",
            "excerpt_note": "Beginning excerpt only; not an exact quoted passage."}
        restored = ca.clean_state(json.loads(json.dumps(state)))
        self.assertEqual(state["sources"]["S1#1"], restored["sources"]["S1#1"])
        markdown = ca.build_handoff(restored, include_sources=True)
        self.assertIn("SOURCE REFERENCE NOTE", markdown)
        self.assertIn("not reconstructed original chat turns", markdown)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = ca.app.test_client()
        turns = []
        for i in range(12):
            turns += [("You", f"message {i} " + "x" * 2800), ("ChatGPT", f"reply {i} " + "y" * 200)]
        self.long_chat = chat(*turns)

    def extract(self, **body):
        return self.client.post("/api/extract", json=body)

    def test_validation(self):
        self.assertEqual(400, self.extract(text="too short").status_code)
        too_long = self.extract(text="You: " + "a" * (ca.MAX_CHARS + 1))
        self.assertEqual(413, too_long.status_code)
        self.assertIn("Nothing was processed", too_long.get_json()["error"])
        self.assertEqual(400, self.extract(text=self.long_chat, mode="gemini").status_code)
        self.assertEqual(400, self.client.post("/api/extract", data="not json", content_type="text/plain").status_code)

    def test_offline_extraction_on_sample(self):
        sample = (Path(ca.__file__).parent / "sample_chat.txt").read_text(encoding="utf-8")
        r = self.extract(text=sample, mode="offline").get_json()
        st = r["state"]
        self.assertEqual("offline", r["report"]["mode"])
        self.assertEqual({"api_key", "email", "password_or_secret"}, set(r["report"]["redactions"]))
        self.assertTrue(any("SQLite" in d["text"] for d in st["decisions"]))
        self.assertEqual([0], r["analysis"]["superseded"])
        self.assertEqual("backend/alerts.py", st["code"][0]["file"])
        self.assertIn("float(heart_rate)", st["code"][0]["code"])
        self.assertNotIn("hunter2x", json.dumps(r))
        self.assertIn("S1#1", st["sources"])
        self.assertEqual("speaker_labelled_turn", st["sources"]["S1#1"]["kind"])

    def test_gemini_path_chunks_merges_and_drops_invented_sources(self):
        with mock.patch.object(ca, "call_gemini", side_effect=fake_gemini()) as m:
            r = self.extract(text=self.long_chat, api_key=FAKE_KEY).get_json()
        self.assertEqual("gemini", r["report"]["mode"])
        self.assertGreaterEqual(r["report"]["chunks"], 3)
        self.assertEqual(r["report"]["chunks"], m.call_count)
        st = r["state"]
        self.assertEqual("Build a monitoring dashboard", st["goal"])
        self.assertEqual(1, len([t for t in st["tech_stack"] if t["text"] == "Flask"]))       # deduped across chunks
        self.assertEqual(1, len([d for d in st["decisions"] if d["text"] == "Use SQLite for storage"]))
        self.assertEqual(r["report"]["chunks"], len(st["bugs"]))                              # distinct bugs kept
        self.assertEqual([], st["code"])                                                      # this chat has no code blocks
        invented = next(d for d in st["decisions"] if d["text"].startswith("Invented"))
        self.assertEqual([], invented["src"])                                                 # unsourced, so the UI flags it
        self.assertNotIn(FAKE_KEY, json.dumps(r))

    def test_bad_chunk_falls_back_with_warning(self):
        good = fake_gemini()

        def flaky(api_key, model, user_text, system=None):
            if "chunk 2 of" in user_text:
                raise ca.ExtractionError("Chunk 2: the model did not return valid JSON (x).")
            return good(api_key, model, user_text)
        with mock.patch.object(ca, "call_gemini", side_effect=flaky):
            r = self.extract(text=self.long_chat, api_key=FAKE_KEY).get_json()
        self.assertTrue(any("offline extractor for messages" in w for w in r["report"]["warnings"]))

    def test_fatal_error_is_reported_without_leaking_key(self):
        err = ca.ExtractionError("Gemini rejected the request (HTTP 403). Check the API key and model name.", fatal=True)
        with mock.patch.object(ca, "call_gemini", side_effect=err):
            resp = self.extract(text=self.long_chat, api_key=FAKE_KEY)
        self.assertEqual(502, resp.status_code)
        self.assertIn("403", resp.get_json()["error"])
        self.assertNotIn(FAKE_KEY, resp.get_data(as_text=True))

    def test_text_sent_to_gemini_is_already_redacted(self):
        seen = []

        def spy(api_key, model, user_text):
            seen.append(user_text)
            return "{}"
        text = chat(("You", f"my password = hunter2x and email bob@example.com, key {FAKE_KEY}"), ("ChatGPT", "ok " * 20))
        with mock.patch.object(ca, "call_gemini", side_effect=spy):
            self.extract(text=text, api_key="k" * 20)
        joined = "\n".join(seen)
        for secret in ["hunter2x", "bob@example.com", FAKE_KEY]:
            self.assertNotIn(secret, joined)

    def test_second_session_merges_into_project_file(self):
        t1 = chat(("You", "I want to build a dashboard. Let's use Firebase for the database."), ("ChatGPT", "Sure, that works well."))
        t2 = chat(("You", "Let's use SQLite instead of Firebase because we need an offline demo."), ("ChatGPT", "Understood."))
        s1 = self.extract(text=t1, mode="offline").get_json()["state"]
        s1["name"] = "Dash"
        r2 = self.extract(text=t2, mode="offline", existing_state=s1).get_json()
        st = r2["state"]
        self.assertEqual((2, 2, "Dash"), (st["version"], st["sessions"], st["name"]))
        self.assertEqual(2, len(st["decisions"]))
        self.assertEqual("S1#1", st["decisions"][0]["src"][0])
        self.assertEqual("S2#1", st["decisions"][1]["src"][0])
        self.assertEqual({"S1#1", "S2#1"}, {k for k in st["sources"] if k in ("S1#1", "S2#1")})
        self.assertEqual([0], r2["analysis"]["superseded"])
        self.assertEqual(st["goal"], s1["goal"])                                              # no new goal was stated

    def test_explicit_project_name_is_extracted_and_merged(self):
        transcript = chat(("You", "Project name: ChatCarry. I want to build a project handoff tool."))
        result = self.extract(text=transcript, mode="offline").get_json()
        self.assertEqual("ChatCarry", result["state"]["name"])

    def test_explicit_new_goal_updates_existing_project_goal(self):
        old = ca.empty_state()
        old.update(name="Demo", goal="Build a basic dashboard", version=1, sessions=1)
        transcript = chat(("You", "I want to change the project goal to build an offline-first project handoff platform."))
        result = self.extract(text=transcript, mode="offline", existing_state=old).get_json()
        self.assertIn("offline-first project handoff platform", result["state"]["goal"])
        self.assertEqual("Demo", result["state"]["name"])

    def test_json_state_roundtrip_and_markdown_export_preserve_evidence(self):
        original = ca.empty_state()
        original.update(name="ChatCarry", goal="Move project context safely between assistants.", version=3)
        original["tech_stack"] = [{"text": "Flask", "status": "approved", "src": ["S1#2"]}]
        original["completed"] = [{"text": "Added a parser", "src": ["S1#3"]}]
        original["sources"]["S1#2"] = {"role": "user", "kind": "speaker_labelled_turn", "label": "User turn",
                                         "text": "We decided to use Flask.",
                                         "excerpt_note": "Source text (message-level reference; not an exact quoted passage)."}
        restored = ca.clean_state(json.loads(json.dumps(original)))
        self.assertEqual(original["tech_stack"], restored["tech_stack"])
        self.assertEqual(original["sources"], restored["sources"])
        markdown = ca.build_handoff(restored, include_sources=True)
        self.assertIn("[Approved; not yet implemented] Flask [S1#2]", markdown)
        self.assertIn("Move project context safely between assistants.", markdown)
        self.assertIn("Added a parser [S1#3]", markdown)

    def test_garbage_state_is_handled(self):
        for bad in [None, "x", 5, [], {"decisions": "no", "code": [1, None, {"code": 5}], "sources": {"x": 1}, "version": "abc"}]:
            r = self.client.post("/api/handoff", json={"state": bad, "target": "nonsense"})
            self.assertEqual(200, r.status_code, bad)
            self.assertIn("PROJECT STATE", r.get_json()["markdown"])
            self.assertEqual(200, self.client.post("/api/conflicts", json={"state": bad}).status_code)

    def test_index_and_sample_routes(self):
        index = self.client.get("/")
        self.assertEqual(200, index.status_code)
        index.close()
        self.assertIn("text", self.client.get("/api/sample").get_json())
        self.assertEqual({"ok": True}, self.client.get("/health").get_json())


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class CallGeminiTests(unittest.TestCase):
    OK = json.dumps({"candidates": [{"content": {"parts": [{"text": "{\"goal\": \"g\"}"}]}}]}).encode()

    def test_request_shape_and_key_in_header_not_url(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["req"] = req
            return FakeResponse(self.OK)
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            out = ca.call_gemini(FAKE_KEY, "gemini-2.5-flash", "hello")
        req = captured["req"]
        self.assertEqual('{"goal": "g"}', out)
        self.assertNotIn(FAKE_KEY, req.full_url)
        self.assertIn("models/gemini-2.5-flash:generateContent", req.full_url)
        self.assertEqual(FAKE_KEY, req.get_header("X-goog-api-key"))
        body = json.loads(req.data)
        self.assertEqual("application/json", body["generationConfig"]["responseMimeType"])
        self.assertIn("systemInstruction", body)

    def test_retries_on_503_then_succeeds(self):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
            return FakeResponse(self.OK)
        with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("time.sleep"):
            self.assertEqual('{"goal": "g"}', ca.call_gemini(FAKE_KEY, "gemini-2.5-flash", "hi"))
        self.assertEqual(2, calls["n"])

    def test_bad_key_is_fatal_and_message_has_no_key(self):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, None)
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(ca.ExtractionError) as cm:
                ca.call_gemini(FAKE_KEY, "gemini-2.5-flash", "hi")
        self.assertTrue(cm.exception.fatal)
        self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_blocked_answer_is_non_fatal(self):
        with mock.patch("urllib.request.urlopen", lambda req, timeout=None: FakeResponse(b'{"candidates": []}')):
            with self.assertRaises(ca.ExtractionError) as cm:
                ca.call_gemini(FAKE_KEY, "gemini-2.5-flash", "hi")
        self.assertFalse(cm.exception.fatal)

    def test_model_name_is_validated(self):
        with self.assertRaises(ca.ExtractionError):
            ca.call_gemini(FAKE_KEY, "x/../../evil?", "hi")

    # ----- speed setting (thinking level) and timeouts -----
    def setUp(self):
        ca._no_thinking_hint.clear()
        self.addCleanup(ca._no_thinking_hint.clear)

    def run_calls(self, behaviours, model="gemini-3.8-flash"):
        """Run call_gemini with a list of per-request behaviours: bytes (answer) or an exception."""
        sent = []

        def fake_urlopen(req, timeout=None):
            sent.append((json.loads(req.data)["generationConfig"], timeout))
            what = behaviours[min(len(sent), len(behaviours)) - 1]
            if isinstance(what, Exception):
                raise what
            return FakeResponse(what)
        with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("time.sleep"):
            try:
                out = ca.call_gemini(FAKE_KEY, model, "hi")
            except ca.ExtractionError as e:
                out = e
        return out, sent

    @staticmethod
    def http(code):
        return urllib.error.HTTPError("https://x", code, "err", {}, None)

    def test_asks_gemini_to_think_less_by_default(self):
        out, sent = self.run_calls([self.OK])
        self.assertEqual('{"goal": "g"}', out)
        self.assertEqual({"thinkingLevel": "low"}, sent[0][0]["thinkingConfig"])
        self.assertEqual(ca.REQUEST_TIMEOUT, sent[0][1])

    def test_falls_back_without_the_setting_and_remembers_it(self):
        out, sent = self.run_calls([self.http(400), self.OK])
        self.assertEqual('{"goal": "g"}', out)
        self.assertEqual(2, len(sent))
        self.assertIn("thinkingConfig", sent[0][0])
        self.assertNotIn("thinkingConfig", sent[1][0])
        self.assertIn("gemini-3.8-flash", ca._no_thinking_hint)
        _, later = self.run_calls([self.OK])
        self.assertNotIn("thinkingConfig", later[0][0])          # not sent again for this model

    def test_a_real_400_is_still_reported_and_not_remembered(self):
        out, sent = self.run_calls([self.http(400), self.http(400)])
        self.assertIsInstance(out, ca.ExtractionError)
        self.assertTrue(out.fatal)
        self.assertIn("HTTP 400", str(out))
        self.assertEqual(2, len(sent))
        self.assertEqual(set(), ca._no_thinking_hint)

    def test_the_setting_can_be_turned_off_or_is_ignored_when_invalid(self):
        for value in ["", "turbo"]:
            with mock.patch.object(ca, "THINKING_LEVEL", value):
                _, sent = self.run_calls([self.OK])
            self.assertNotIn("thinkingConfig", sent[0][0], value)
        with mock.patch.object(ca, "THINKING_LEVEL", "minimal"):
            _, sent = self.run_calls([self.OK])
        self.assertEqual({"thinkingLevel": "minimal"}, sent[0][0]["thinkingConfig"])

    def test_timeout_gets_one_retry_and_an_honest_message(self):
        out, sent = self.run_calls([TimeoutError("timed out")])
        self.assertIsInstance(out, ca.ExtractionError)
        self.assertTrue(out.fatal)
        self.assertEqual(2, len(sent))
        self.assertIn("did not answer within 120 seconds", str(out))
        self.assertNotIn("internet", str(out))

    def test_a_connect_timeout_wrapped_in_urlerror_counts_as_a_timeout(self):
        out, sent = self.run_calls([urllib.error.URLError(TimeoutError("timed out"))])
        self.assertIn("did not answer", str(out))
        self.assertEqual(2, len(sent))

    def test_a_slow_first_answer_then_success(self):
        out, sent = self.run_calls([TimeoutError("timed out"), self.OK])
        self.assertEqual('{"goal": "g"}', out)
        self.assertEqual(2, len(sent))

    def test_no_connection_is_still_called_a_connection_problem(self):
        out, sent = self.run_calls([urllib.error.URLError("name resolution failed")])
        self.assertIn("internet connection", str(out))
        self.assertEqual(4, len(sent))


def make_long_chat():
    turns = []
    for i in range(12):
        turns += [("You", f"message {i} " + "x" * 2800), ("ChatGPT", f"reply {i} " + "y" * 200)]
    return chat(*turns)


class EnvFileTests(unittest.TestCase):
    def test_loads_values_and_real_environment_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_bytes(b'\xef\xbb\xbf# comment\nCC_TEST_KEY="abc123"\nexport CC_TEST_B=two # note\n'
                          b"CC_TEST_C=\nthis line is junk\nCC_TEST_A=fromfile\nCC_TEST_D='x y'\n")
            with mock.patch.dict(os.environ, {"CC_TEST_A": "real"}):
                loaded = ca.load_env_file(p)
                self.assertEqual("abc123", os.environ["CC_TEST_KEY"])
                self.assertEqual("two", os.environ["CC_TEST_B"])
                self.assertEqual("x y", os.environ["CC_TEST_D"])
                self.assertEqual("real", os.environ["CC_TEST_A"])
                self.assertNotIn("CC_TEST_C", os.environ)
                self.assertEqual({"CC_TEST_KEY", "CC_TEST_B", "CC_TEST_D"}, set(loaded))
        self.assertEqual([], ca.load_env_file("/no/such/folder/.env"))


class ConfigAndHostTests(unittest.TestCase):
    def setUp(self):
        self.client = ca.app.test_client()

    def test_config_reports_a_key_without_revealing_it(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "SECRETVALUE123"}):
            r = self.client.get("/api/config")
        self.assertTrue(r.get_json()["has_env_key"])
        self.assertNotIn("SECRETVALUE123", r.get_data(as_text=True))
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertFalse(self.client.get("/api/config").get_json()["has_env_key"])

    def test_env_key_is_used_when_none_is_typed(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": FAKE_KEY}), \
                mock.patch.object(ca, "call_gemini", side_effect=fake_gemini()) as m:
            r = self.client.post("/api/extract", json={"text": make_long_chat()}).get_json()
        self.assertEqual("gemini", r["report"]["mode"])
        self.assertEqual(FAKE_KEY, m.call_args_list[0][0][0])
        self.assertNotIn(FAKE_KEY, json.dumps(r))

    def test_other_host_names_are_refused(self):
        for host, code in [("localhost:5000", 200), ("127.0.0.1:5000", 200), ("[::1]:5000", 200),
                           ("evil.example.com", 403), ("evil.example.com:5000", 403), ("127.0.0.1.evil.com", 403)]:
            r = self.client.get("/health", headers={"Host": host})
            self.assertEqual(code, r.status_code, host)


BIG_CODE = "\n".join(f"    line_{i} = {i}  # keep exactly" for i in range(60))


class CodeKeptLocallyTests(unittest.TestCase):
    def setUp(self):
        self.client = ca.app.test_client()
        self.text = chat(
            ("You", "Build a tool. Please write main.py"),
            ("ChatGPT", "Here is main.py:\n```python\ndef run():\n" + BIG_CODE + "\n```\nRun `pip install flask` first.\n```bash\npip install flask\n```"),
            ("You", "It crashes with an error on line 3. Fix it please"),
            ("ChatGPT", "Fixed version of main.py:\n```python\ndef run():\n    return 2\n    # second version\n```"),
        )

    def run_gemini(self, fake=None):
        with mock.patch.object(ca, "call_gemini", side_effect=fake or fake_gemini()) as m:
            r = self.client.post("/api/extract", json={"text": self.text, "api_key": FAKE_KEY}).get_json()
        return r, m

    def test_prompt_has_only_a_stub_of_big_code(self):
        r, m = self.run_gemini()
        prompt = m.call_args_list[0].args[2]
        self.assertIn("[code block C1", prompt)
        self.assertIn("line_0 = 0", prompt)
        self.assertNotIn("line_40 = 40", prompt)
        self.assertIn("more lines not shown", prompt)
        self.assertIn("```bash\npip install flask\n```", prompt)   # tiny shell commands stay inline

    def test_code_is_word_for_word_and_model_code_is_ignored(self):
        r, _ = self.run_gemini()
        code = r["state"]["code"]
        self.assertEqual(1, len(code))                            # newest version of main.py wins
        self.assertEqual(2, code[0]["versions"])
        self.assertIn("second version", code[0]["code"])
        self.assertNotIn("TAMPERED", json.dumps(r))
        self.assertEqual("ok", code[0]["status"])                # status comes from the model
        self.assertEqual("main.py", code[0]["file"])             # the chat's own file name wins

    def test_big_block_is_kept_in_full_when_it_is_the_only_version(self):
        self.text = chat(("You", "write main.py"), ("ChatGPT", "main.py:\n```python\ndef run():\n" + BIG_CODE + "\n```"))
        r, _ = self.run_gemini()
        self.assertIn("line_59 = 59  # keep exactly", r["state"]["code"][0]["code"])

    def test_fallback_path_does_not_duplicate_code(self):
        good = fake_gemini()

        def flaky(api_key, model, user_text, system=None):
            raise ca.ExtractionError("Chunk 1: the model did not return valid JSON (x).")
        r, _ = self.run_gemini(flaky)
        self.assertEqual(1, len(r["state"]["code"]))
        self.assertEqual("unknown", r["state"]["code"][0]["status"])
        self.assertTrue(r["report"]["warnings"])


class PingAndTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.client = ca.app.test_client()

    def test_ping_ok(self):
        with mock.patch.object(ca, "call_gemini", return_value="{}") as m:
            r = self.client.post("/api/gemini_test", json={"api_key": FAKE_KEY, "model": "gemini-x-flash-lite"})
        d = r.get_json()
        self.assertTrue(d["ok"])
        self.assertEqual("gemini-x-flash-lite", d["model"])
        self.assertIn("seconds", d)
        self.assertNotIn(FAKE_KEY, json.dumps(d))

    def test_ping_without_key_and_with_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(400, self.client.post("/api/gemini_test", json={}).status_code)
        with mock.patch.object(ca, "call_gemini", side_effect=ca.ExtractionError("nope", fatal=True)):
            r = self.client.post("/api/gemini_test", json={"api_key": FAKE_KEY})
        self.assertEqual(502, r.status_code)
        self.assertEqual("nope", r.get_json()["error"])

    def test_timeout_error_names_the_part(self):
        def slow(api_key, model, user_text, system=None):
            raise ca.ExtractionError("Gemini did not answer.", fatal=True, timeout=True)
        with mock.patch.object(ca, "call_gemini", side_effect=slow):
            r = self.client.post("/api/extract", json={"text": make_long_chat(), "api_key": FAKE_KEY, "model": "m-flash"})
        self.assertEqual(502, r.status_code)
        msg = r.get_json()["error"]
        self.assertRegex(msg, r"\(Part 1 of \d+, [\d,]+ characters, model m-flash\.\)")


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.client = ca.app.test_client()

    def events(self, response):
        return [json.loads(line) for line in response.get_data(as_text=True).splitlines()]

    def test_progress_events_then_result(self):
        with mock.patch.object(ca, "call_gemini", side_effect=fake_gemini()):
            r = self.client.post("/api/extract_stream", json={"text": make_long_chat(), "api_key": FAKE_KEY}, buffered=True)
        self.assertEqual("application/x-ndjson", r.mimetype)
        ev = self.events(r)
        self.assertEqual("result", ev[-1]["type"])
        self.assertEqual("progress", ev[0]["type"])
        chunks = ev[-1]["report"]["chunks"]
        extract = [e for e in ev if e["type"] == "progress" and e["stage"] == "extract"]
        self.assertEqual(2 * chunks, len(extract))
        self.assertEqual(sorted(e["done"] for e in extract), [e["done"] for e in sorted(extract, key=lambda e: e["done"])])
        self.assertEqual((chunks, chunks), (extract[-1]["done"], extract[-1]["total"]))
        self.assertEqual(1, len([e for e in ev if e["type"] == "result"]))
        self.assertNotIn(FAKE_KEY, r.get_data(as_text=True))

    def test_stream_matches_normal_endpoint(self):
        with mock.patch.object(ca, "call_gemini", side_effect=fake_gemini()):
            body = {"text": make_long_chat(), "api_key": FAKE_KEY}
            streamed = self.events(self.client.post("/api/extract_stream", json=body, buffered=True))[-1]
            normal = self.client.post("/api/extract", json=body).get_json()
        streamed.pop("type")
        self.assertEqual(normal, streamed)

    def test_errors_arrive_as_events(self):
        ev = self.events(self.client.post("/api/extract_stream", json={"text": "short"}))
        self.assertEqual([("error", 400)], [(e["type"], e["status"]) for e in ev])
        err = ca.ExtractionError("Gemini rejected the request (HTTP 403).", fatal=True)
        with mock.patch.object(ca, "call_gemini", side_effect=err):
            ev = self.events(self.client.post("/api/extract_stream", json={"text": make_long_chat(), "api_key": FAKE_KEY}, buffered=True))
        self.assertEqual(("error", 502), (ev[-1]["type"], ev[-1]["status"]))

    def test_unexpected_crash_becomes_a_plain_error_event(self):
        with mock.patch.object(ca, "redact", side_effect=RuntimeError("boom-secret-detail")):
            r = self.client.post("/api/extract_stream", json={"text": make_long_chat()}, buffered=True)
        ev = self.events(r)
        self.assertEqual(("error", 500), (ev[-1]["type"], ev[-1]["status"]))
        self.assertNotIn("boom-secret-detail", r.get_data(as_text=True))


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "projects"
        patcher = mock.patch.object(ca, "PROJECTS_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.client = ca.app.test_client()

    def state(self, name="Story DNA!", version=1):
        st = ca.empty_state()
        st.update(name=name, version=version, sessions=1, updated="2026-10-28", goal="Make videos")
        st["decisions"] = [{"text": "Use Python", "src": ["S1#1"], "reason": "", "topic": "", "choice": ""}]
        st["bugs"] = [{"text": "open one", "src": [], "status": "open"}, {"text": "done one", "src": [], "status": "fixed"}]
        return st

    def save(self, **body):
        return self.client.post("/api/projects", json=body)

    def test_save_list_get_overwrite_delete(self):
        r = self.save(state=self.state())
        self.assertEqual(200, r.status_code)
        pid = r.get_json()["id"]
        self.assertRegex(pid, r"^story-dna-[0-9a-f]{6}$")
        listing = self.client.get("/api/projects").get_json()["projects"]
        self.assertEqual(1, len(listing))
        self.assertEqual({"decisions": 1, "pending": 0, "open_bugs": 1, "code": 0}, listing[0]["counts"])
        self.assertEqual("Story DNA!", listing[0]["name"])
        self.assertEqual("Make videos", self.client.get(f"/api/projects/{pid}").get_json()["state"]["goal"])

        again = self.save(id=pid, state=self.state(version=2)).get_json()
        self.assertEqual(pid, again["id"])
        self.assertEqual(1, len(self.client.get("/api/projects").get_json()["projects"]))
        self.assertEqual(2, self.client.get(f"/api/projects/{pid}").get_json()["state"]["version"])
        self.assertEqual([], list(self.dir.glob("*.tmp")))

        self.assertEqual(200, self.client.delete(f"/api/projects/{pid}").status_code)
        self.assertEqual(404, self.client.get(f"/api/projects/{pid}").status_code)
        self.assertEqual(404, self.client.delete(f"/api/projects/{pid}").status_code)
        self.assertEqual([], self.client.get("/api/projects").get_json()["projects"])

    def test_a_name_is_required(self):
        self.assertEqual(400, self.save(state=self.state(name="")).status_code)
        self.assertEqual(400, self.save(state="garbage").status_code)

    def test_unsafe_or_unknown_ids_never_touch_other_files(self):
        outside = Path(self.tmp.name) / "secret.json"
        outside.write_text('{"state": {"name": "secret"}}')
        for bad in ["..%2Fsecret", "..", "a/b", "A_B", "x" * 100, "-start", ""]:
            self.assertEqual(404, self.client.get(f"/api/projects/{bad}").status_code, bad)
            self.assertIn(self.client.delete(f"/api/projects/{bad}").status_code, (404, 405), bad)
        self.assertTrue(outside.exists())
        r = self.save(id="../secret", state=self.state()).get_json()
        self.assertNotEqual("../secret", r["id"])
        self.assertEqual('{"state": {"name": "secret"}}', outside.read_text())
        self.assertEqual(1, len(list(self.dir.glob("*.json"))))
        r2 = self.save(id="valid-but-missing", state=self.state()).get_json()
        self.assertNotEqual("valid-but-missing", r2["id"])

    def test_broken_files_are_skipped(self):
        self.dir.mkdir(parents=True)
        (self.dir / "broken.json").write_text("{not json")
        (self.dir / "list.json").write_text("[1, 2]")
        (self.dir / "Bad Name.json").write_text("{}")
        self.save(state=self.state())
        self.assertEqual(1, len(self.client.get("/api/projects").get_json()["projects"]))
        self.assertEqual(404, self.client.get("/api/projects/broken").status_code)

    def test_saved_state_is_cleaned(self):
        st = self.state()
        st["decisions"][0]["evil"] = "<script>"
        st["unknown_key"] = "x"
        pid = self.save(state=st).get_json()["id"]
        saved = json.loads((self.dir / f"{pid}.json").read_text())["state"]
        self.assertNotIn("unknown_key", saved)
        self.assertNotIn("evil", saved["decisions"][0])

    def test_project_limit(self):
        with mock.patch.object(ca, "MAX_PROJECTS", 2):
            self.assertEqual(200, self.save(state=self.state("one")).status_code)
            self.assertEqual(200, self.save(state=self.state("two")).status_code)
            self.assertEqual(400, self.save(state=self.state("three")).status_code)
            pid = self.client.get("/api/projects").get_json()["projects"][0]["id"]
            self.assertEqual(200, self.save(id=pid, state=self.state("one again")).status_code)


if __name__ == "__main__":
    unittest.main()
