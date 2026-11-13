/* ChatCarry: read the conversations.json file from a ChatGPT or Claude data export.
 *
 * Runs in the browser, so a big export never has to be uploaded just to pick one chat.
 *
 *   const exp = ChatCarryImport.parseExport(fileText);
 *   exp.source          -> 'chatgpt' or 'claude'
 *   exp.conversations   -> [{ index, id, title, updated (ms or null), messages }]  newest first
 *   exp.transcript(i)   -> "You: ...\n\nChatGPT: ..." for conversations[i].index
 */
(function (root) {
  'use strict';

  const ROLE_LINE = /^\s*(?:#{1,4}\s*)?\**\s*(?:You|User|Human|Me|ChatGPT|Assistant|Claude|Gemini|AI|Bot|Copilot|Grok|DeepSeek|Perplexity)\s*(?:said)?\s*\**\s*[:：]/i;
  const ZWSP = '​';
  const LABELS = {
    chatgpt: { user: 'You', assistant: 'ChatGPT' },
    claude: { user: 'You', assistant: 'Claude' },
  };

  function isObject(x) { return x !== null && typeof x === 'object' && !Array.isArray(x); }
  function toMs(value) {
    if (typeof value === 'number' && isFinite(value)) return value < 1e11 ? value * 1000 : value; // seconds or ms
    if (typeof value === 'string') { const t = Date.parse(value); return isNaN(t) ? null : t; }
    return null;
  }

  // A line inside a message that looks like "User: ..." would be mistaken for a new speaker
  // by the splitter, so an invisible character is put in front of it. Code blocks stay untouched.
  function protectRoleLines(text) {
    let inFence = false;
    return text.split('\n').map(line => {
      if (line.trim().startsWith('```')) { inFence = !inFence; return line; }
      return !inFence && ROLE_LINE.test(line) ? ZWSP + line : line;
    }).join('\n');
  }

  // ---------- ChatGPT ----------
  const GPT_CONTENT_TYPES = ['text', 'multimodal_text', 'code'];

  function gptPath(conv) {
    const map = conv.mapping;
    let id = conv.current_node;
    if (!id || !isObject(map[id])) {            // no current node: use the newest leaf
      let best = null, bestTime = -1;
      for (const key of Object.keys(map)) {
        const node = map[key];
        if (!isObject(node)) continue;
        const kids = Array.isArray(node.children) ? node.children : [];
        if (kids.length === 0) {
          const t = (node.message && typeof node.message.create_time === 'number') ? node.message.create_time : 0;
          if (best === null || t >= bestTime) { best = key; bestTime = t; }
        }
      }
      id = best;
    }
    const path = [], seen = new Set();
    while (id && isObject(map[id]) && !seen.has(id)) {
      seen.add(id);
      path.push(map[id]);
      id = map[id].parent;
    }
    return path.reverse();
  }

  function gptText(message) {
    const c = message.content;
    if (!isObject(c) || !GPT_CONTENT_TYPES.includes(c.content_type)) return '';
    const out = [];
    if (Array.isArray(c.parts)) {
      for (const part of c.parts) {
        if (typeof part === 'string') { if (part.trim()) out.push(part); }
        else if (isObject(part)) {
          if (part.content_type === 'image_asset_pointer') out.push('[image]');
          else if (typeof part.text === 'string' && part.text.trim()) out.push(part.text);
        }
      }
    } else if (typeof c.text === 'string' && c.text.trim()) {
      out.push(c.text);
    }
    let text = out.join('\n\n').trim();
    if (text && c.content_type === 'code') {
      const lang = typeof c.language === 'string' && c.language !== 'unknown' ? c.language : '';
      text = '```' + lang + '\n' + text + '\n```';
    }
    return text;
  }

  function gptMessages(conv) {
    const out = [];
    for (const node of gptPath(conv)) {
      const m = node.message;
      if (!isObject(m) || !isObject(m.author)) continue;
      const role = m.author.role;
      if (role !== 'user' && role !== 'assistant') continue;        // skip system and tool output
      if (isObject(m.metadata) && m.metadata.is_visually_hidden_from_conversation === true) continue;
      if (role === 'assistant' && m.recipient && m.recipient !== 'all') continue; // tool calls
      const text = gptText(m);
      if (text) out.push({ role, text });
    }
    return out;
  }

  // ---------- Claude ----------
  function claudeText(m) {
    const parts = [];
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (isObject(block) && block.type === 'text' && typeof block.text === 'string' && block.text.trim()) parts.push(block.text);
      }
    }
    if (!parts.length && typeof m.text === 'string' && m.text.trim()) parts.push(m.text);
    for (const list of [m.attachments, m.files]) {
      for (const a of Array.isArray(list) ? list : []) {
        if (isObject(a) && typeof a.file_name === 'string' && a.file_name) parts.push('[Attached file: ' + a.file_name + ']');
      }
    }
    return parts.join('\n\n').trim();
  }

  function claudeMessages(conv) {
    const out = [];
    for (const m of conv.chat_messages) {
      if (!isObject(m)) continue;
      const role = m.sender === 'human' ? 'user' : m.sender === 'assistant' ? 'assistant' : null;
      if (!role) continue;
      const text = claudeText(m);
      if (text) out.push({ role, text });
    }
    return out;
  }

  // ---------- public ----------
  function parseExport(fileText) {
    let data;
    try { data = JSON.parse(fileText); }
    catch (e) { throw new Error('That file is not valid JSON. Pick the conversations.json file from your export (unzip it first).'); }

    let list = [];
    if (Array.isArray(data)) list = data;
    else if (isObject(data) && Array.isArray(data.conversations)) list = data.conversations;
    else if (isObject(data)) list = [data];
    list = list.filter(isObject);

    let source = null;
    if (list.some(c => isObject(c.mapping))) source = 'chatgpt';
    else if (list.some(c => Array.isArray(c.chat_messages))) source = 'claude';
    if (!source) {
      throw new Error('This does not look like a ChatGPT or Claude export. Use the conversations.json file from the data export (unzip it first).');
    }

    const valid = list.filter(c => source === 'chatgpt' ? isObject(c.mapping) : Array.isArray(c.chat_messages));
    const messagesOf = source === 'chatgpt' ? gptMessages : claudeMessages;
    const labels = LABELS[source];

    const conversations = valid.map((c, index) => ({
      index,
      id: String(c.id || c.conversation_id || c.uuid || index),
      title: String((source === 'chatgpt' ? c.title : (c.name || c.title)) || '').trim() || 'Untitled chat',
      updated: toMs(source === 'chatgpt' ? (c.update_time || c.create_time) : (c.updated_at || c.created_at)),
      messages: messagesOf(c).length,
    })).filter(c => c.messages > 0);
    conversations.sort((a, b) => (b.updated || 0) - (a.updated || 0));
    if (!conversations.length) throw new Error('No readable messages were found in that file.');

    return {
      source,
      conversations,
      transcript(index) {
        const conv = valid[index];
        if (!conv) throw new Error('Conversation not found.');
        return messagesOf(conv).map(m => labels[m.role] + ': ' + protectRoleLines(m.text)).join('\n\n');
      },
    };
  }

  const api = { parseExport, protectRoleLines };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ChatCarryImport = api;
})(typeof window !== 'undefined' ? window : globalThis);
