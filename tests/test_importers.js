// Tests for static/importers.js.  Run from the project folder with:  node --test tests/
// The export files below are hand-made copies of the ChatGPT and Claude formats.
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { parseExport, protectRoleLines } = require('../static/importers.js');

function gptNode(id, parent, children, role, parts, extra = {}) {
  return {
    id, parent, children,
    message: role === null ? null : {
      id, author: { role }, create_time: extra.time || 1,
      content: extra.content || { content_type: 'text', parts },
      metadata: extra.metadata || {}, recipient: extra.recipient || 'all',
    },
  };
}

function gptConversation() {
  return {
    title: 'Story project', create_time: 1700000000, update_time: 1700000500, current_node: 'a2',
    mapping: {
      root: gptNode('root', null, ['sys'], null),
      sys: gptNode('sys', 'root', ['u1'], 'system', ['You are ChatGPT']),
      u1: gptNode('u1', 'sys', ['a1', 'a1b'], 'user', ['Build a story generator.']),
      a1b: gptNode('a1b', 'u1', [], 'assistant', ['OLD BRANCH that was regenerated']),
      a1: gptNode('a1', 'u1', ['tool', 'hidden'], 'assistant', ['Use Python.', { content_type: 'image_asset_pointer' }]),
      tool: gptNode('tool', 'a1', ['call'], 'tool', ['tool output should not show']),
      call: gptNode('call', 'tool', ['hidden'], 'assistant', [''], {
        recipient: 'python', content: { content_type: 'code', language: 'python', text: 'scratch = 1' },
      }),
      hidden: gptNode('hidden', 'call', ['u2'], 'user', ['hidden context'], { metadata: { is_visually_hidden_from_conversation: true } }),
      u2: gptNode('u2', 'hidden', ['a2'], 'user', ['Show me the main file.']),
      a2: gptNode('a2', 'u2', [], 'assistant', ['Here it is:\nUser: not a speaker\n```python\nuser: str = "x"\n```']),
    },
  };
}

function claudeConversation() {
  return {
    uuid: 'c-1', name: 'Video pipeline', created_at: '2026-10-01T10:00:00Z', updated_at: '2026-10-05T10:00:00Z',
    chat_messages: [
      { uuid: 'm1', sender: 'human', text: 'ignored because content exists', created_at: '2026-10-01T10:00:00Z',
        content: [{ type: 'text', text: 'Make a video from a story.' }],
        attachments: [{ file_name: 'notes.txt', extracted_content: 'long text that must not be copied' }] },
      { uuid: 'm2', sender: 'assistant', text: 'Plan',
        content: [{ type: 'thinking', thinking: 'secret thoughts' }, { type: 'text', text: 'Use FFmpeg.' }, { type: 'tool_use', name: 'x' }] },
      { uuid: 'm3', sender: 'human', text: 'Thanks, next we need subtitles.', content: [] },
      { uuid: 'm4', sender: 'assistant', text: '', content: [{ type: 'tool_use', name: 'x' }] },
      { uuid: 'm5', sender: 'system', text: 'ignore me' },
    ],
  };
}

test('ChatGPT: follows the active branch and keeps only real messages', () => {
  const exp = parseExport(JSON.stringify([gptConversation()]));
  assert.strictEqual(exp.source, 'chatgpt');
  assert.deepStrictEqual(exp.conversations.map(c => [c.title, c.messages]), [['Story project', 4]]);
  const t = exp.transcript(exp.conversations[0].index);
  assert.match(t, /^You: Build a story generator\.\n\nChatGPT: Use Python\.\n\n\[image\]\n\nYou: Show me the main file\.\n\nChatGPT: Here it is:/);
  for (const bad of ['OLD BRANCH', 'tool output', 'scratch = 1', 'hidden context', 'You are ChatGPT']) {
    assert.ok(!t.includes(bad), bad);
  }
});

test('ChatGPT: speaker-looking lines are protected, code is left alone', () => {
  const exp = parseExport(JSON.stringify([gptConversation()]));
  const t = exp.transcript(0);
  assert.ok(t.includes('​User: not a speaker'));
  assert.ok(t.includes('\nuser: str = "x"\n'));
  assert.ok(!t.includes('​user: str'));
});

test('ChatGPT: the real splitter sees exactly the right messages', (ctx) => {
  const exp = parseExport(JSON.stringify([gptConversation()]));
  const t = exp.transcript(0);
  const root = path.join(__dirname, '..');
  const py = spawnSync('python3', ['-c',
    'import sys, json; sys.path.insert(0, sys.argv[1]); import app; ' +
    'print(json.dumps([[m["role"], m["text"][:12]] for m in app.split_messages(sys.stdin.read())]))', root],
  { input: t, encoding: 'utf8' });
  if (py.error || py.status !== 0) { ctx.skip('python3 not available'); return; }
  assert.deepStrictEqual(JSON.parse(py.stdout), [['user', 'Build a stor'], ['assistant', 'Use Python.\n'], ['user', 'Show me the '], ['assistant', 'Here it is:\n']]);
});

test('ChatGPT: without current_node it uses the newest leaf; a parent loop does not hang', () => {
  const conv = gptConversation();
  delete conv.current_node;
  conv.mapping.a2.message.create_time = 99;
  const exp = parseExport(JSON.stringify([conv]));
  assert.match(exp.transcript(0), /Show me the main file/);

  const loop = gptConversation();
  loop.mapping.u1.parent = 'a2';           // a cycle
  const out = parseExport(JSON.stringify([loop])).transcript(0);
  assert.ok(out.length > 0);
});

test('Claude: reads text blocks, skips thinking and tools, notes attachments', () => {
  const exp = parseExport(JSON.stringify([claudeConversation()]));
  assert.strictEqual(exp.source, 'claude');
  assert.deepStrictEqual(exp.conversations.map(c => [c.id, c.title, c.messages]), [['c-1', 'Video pipeline', 3]]);
  const t = exp.transcript(0);
  assert.strictEqual(t,
    'You: Make a video from a story.\n\n[Attached file: notes.txt]\n\nClaude: Use FFmpeg.\n\nYou: Thanks, next we need subtitles.');
  for (const bad of ['secret thoughts', 'long text that must not', 'ignore me', 'ignored because']) assert.ok(!t.includes(bad), bad);
});

test('conversations are listed newest first, empty ones dropped, untitled named', () => {
  const a = claudeConversation(); a.uuid = 'old'; a.name = 'Old'; a.updated_at = '2026-01-01T00:00:00Z';
  const b = claudeConversation(); b.uuid = 'new'; b.name = ''; b.updated_at = '2026-12-01T00:00:00Z';
  const empty = claudeConversation(); empty.uuid = 'empty'; empty.chat_messages = [];
  const exp = parseExport(JSON.stringify([a, empty, b]));
  assert.deepStrictEqual(exp.conversations.map(c => [c.id, c.title]), [['new', 'Untitled chat'], ['old', 'Old']]);
  assert.match(exp.transcript(exp.conversations[0].index), /^You: Make a video/);
});

test('wrapped and single-conversation files are accepted', () => {
  assert.strictEqual(parseExport(JSON.stringify({ conversations: [claudeConversation()] })).conversations.length, 1);
  assert.strictEqual(parseExport(JSON.stringify(gptConversation())).conversations.length, 1);
});

test('bad files give a clear message', () => {
  assert.throws(() => parseExport('not json'), /not valid JSON/);
  assert.throws(() => parseExport('[]'), /does not look like/);
  assert.throws(() => parseExport('{"a": 1}'), /does not look like/);
  assert.throws(() => parseExport('"text"'), /does not look like/);
  assert.throws(() => parseExport(JSON.stringify([{ chat_messages: [] }])), /No readable messages/);
});

test('protectRoleLines only touches speaker-like lines outside code', () => {
  const out = protectRoleLines('hello\nHuman: hi\n  **Claude**: yes\nassistant said: x\n```\nai: y\n```\nnormal: line');
  assert.strictEqual(out, 'hello\n​Human: hi\n​  **Claude**: yes\n​assistant said: x\n```\nai: y\n```\nnormal: line');
});
