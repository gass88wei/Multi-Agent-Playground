#!/usr/bin/env node

/**
 * SnapDesign RedNote v2
 * - Primary mode: LLM -> structured JSON plan -> deterministic premium templates.
 * - Backup mode: LLM direct HTML with strict retry.
 * - Final fallback: static template from source text.
 */

const fs = require('fs');
const path = require('path');

const LLM_API_KEY = process.env.LLM_API_KEY || '';
const LLM_BASE_URL = (process.env.LLM_BASE_URL || 'https://openrouter.ai/api/v1').replace(/\/+$/, '');
const LLM_HTML_MODEL = process.env.LLM_MODEL || 'anthropic/claude-3.5-sonnet';
const LLM_IMAGE_MODEL = process.env.LLM_IMAGE_MODEL || 'seedream-4.5';

const COLORS = {
  background: '#FFFCF8',
  text: '#5A3E35',
  textDark: '#2D1E19',
  accent1: '#D4A574',
  accent2: '#E8B4A0',
  accent3: '#A8B5A0',
  topBar: '#6F4A42',
};

const KICKERS = ['STAGE OVERVIEW', 'CORE TAKEAWAY', 'ACTION GUIDE', 'KEY INSIGHT', 'EXECUTION PLAN'];

function stripMarkdownFences(text) {
  const source = String(text || '').trim();
  return source.replace(/^\s*```(?:json|html)?\s*/i, '').replace(/\s*```\s*$/i, '').trim();
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function clipText(text, maxLen) {
  const source = normalizeText(text);
  if (source.length <= maxLen) return source;
  return `${source.slice(0, maxLen - 1)}…`;
}

function splitTextBlocks(text, targetCount = 3) {
  const source = normalizeText(text);
  if (!source) return ['No content provided.'];

  const chunks = source
    .split(/\n+|(?<=[。！？!?\.])/)
    .map((item) => normalizeText(item))
    .filter(Boolean);
  if (chunks.length <= targetCount) return chunks;

  const merged = [];
  const per = Math.ceil(chunks.length / targetCount);
  for (let i = 0; i < chunks.length; i += per) {
    merged.push(chunks.slice(i, i + per).join(' '));
  }
  return merged;
}

function buildStructuredPlanPrompt(text, cardCount, title = '') {
  return `You are generating content structure for Xiaohongshu cards.

Input content:
"${text}"

Task:
- Create exactly ${cardCount} cards.
- Return STRICT JSON only (no markdown, no explanation).
- Keep user's language.
- Extract and rewrite concise, high-value content.
- Avoid fluff and avoid repeating the same sentence.

JSON schema:
{
  "title": "string",
  "cards": [
    {
      "headline": "string",
      "lead": "string",
      "callout": "string",
      "bullets": ["string", "string", "string", "string"]
    }
  ]
}

Constraints:
- cards length must be exactly ${cardCount}.
- headline: 8-22 chars preferred for Chinese (or concise in other languages).
- lead: 40-110 chars.
- callout: 20-55 chars.
- bullets: 3-5 items, each 10-32 chars preferred.
- No emoji.
- Keep title relevant. If missing, derive one.

User-provided title (if any): "${title || ''}"`;
}

function getPlanSystemInstruction() {
  return 'You are a content editor. Return valid JSON only. No markdown fences.';
}

function parseJsonPayload(raw) {
  const source = String(raw || '').trim();
  if (!source) return null;

  const candidates = [];
  const fenceMatch = source.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) candidates.push(fenceMatch[1].trim());

  candidates.push(stripMarkdownFences(source));

  const firstBrace = source.indexOf('{');
  const lastBrace = source.lastIndexOf('}');
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(source.slice(firstBrace, lastBrace + 1));
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      // try next
    }
  }
  return null;
}

function normalizeBulletList(bullets, fallback) {
  const source = Array.isArray(bullets) ? bullets : [];
  const cleaned = source
    .map((item) => clipText(item, 40))
    .map((item) => normalizeText(item))
    .filter(Boolean);
  if (cleaned.length >= 3) return cleaned.slice(0, 5);

  const fallbackBullets = splitTextBlocks(fallback, 4).map((item) => clipText(item, 36));
  while (fallbackBullets.length < 3) {
    fallbackBullets.push('提炼关键点并形成可执行动作');
  }
  return fallbackBullets.slice(0, 5);
}

function normalizeCardPlan(parsedPlan, sourceText, title, cardCount) {
  const fallbackBlocks = splitTextBlocks(sourceText, cardCount);
  const cardsInput = Array.isArray(parsedPlan?.cards) ? parsedPlan.cards : [];
  const normalizedCards = [];

  for (let i = 0; i < cardCount; i += 1) {
    const fromPlan = cardsInput[i] || {};
    const fallback = fallbackBlocks[i] || fallbackBlocks[fallbackBlocks.length - 1] || sourceText;
    const headline = clipText(fromPlan.headline || fromPlan.title || `第${i + 1}部分`, 28);
    const lead = clipText(fromPlan.lead || fallback, 150);
    const callout = clipText(fromPlan.callout || `核心观点：${fallback}`, 75);
    const bullets = normalizeBulletList(fromPlan.bullets, fallback);

    normalizedCards.push({
      stage: String(i + 1).padStart(2, '0'),
      kicker: KICKERS[i % KICKERS.length],
      headline,
      lead,
      callout,
      bullets,
    });
  }

  const normalizedTitle = clipText(parsedPlan?.title || title || '主题拆解', 30);
  return { title: normalizedTitle, cards: normalizedCards };
}

function headingClassByLength(headline) {
  const len = normalizeText(headline).length;
  if (len <= 10) return 'text-[86px]';
  if (len <= 16) return 'text-[76px]';
  if (len <= 24) return 'text-[66px]';
  return 'text-[58px]';
}

function leadClassByLength(lead) {
  const len = normalizeText(lead).length;
  if (len <= 36) return 'text-[46px] leading-[1.4]';
  if (len <= 70) return 'text-[40px] leading-[1.45]';
  return 'text-[34px] leading-[1.5]';
}

function renderCardTemplateA(card) {
  const headingClass = headingClassByLength(card.headline);
  const leadClass = leadClassByLength(card.lead);
  const bullets = card.bullets
    .map((item) => `<li class="text-[32px] leading-[1.45] text-[#3E2723] whitespace-normal break-all">${escapeHtml(item)}</li>`)
    .join('');

  return `<div class="relative w-[900px] h-[1198px] bg-[#FFFCF8] flex flex-col justify-start py-[84px] px-[76px] overflow-hidden shadow-lg border border-neutral-200 rounded-[26px]">
  <div class="absolute top-0 left-0 right-0 h-[14px] bg-[#6F4A42]"></div>
  <div class="flex items-center justify-between mt-2">
    <div class="w-[72px] h-[72px] rounded-full bg-[#6F4A42] text-white text-[31px] font-bold flex items-center justify-center">${escapeHtml(card.stage)}</div>
    <div class="text-[24px] tracking-[0.18em] uppercase text-[#8F756A]">${escapeHtml(card.kicker)}</div>
  </div>
  <div class="mt-8 flex flex-col gap-7">
    <h1 class="font-serif ${headingClass} leading-[1.08] text-[#2D1E19] whitespace-normal break-all">${escapeHtml(card.headline)}</h1>
    <p class="${leadClass} text-[#5A3E35] whitespace-normal break-all">${escapeHtml(card.lead)}</p>
    <div class="rounded-[26px] border border-[#D4A574]/40 bg-[#E8B4A0]/32 p-7">
      <p class="text-[33px] leading-[1.45] font-semibold text-[#3E2723] whitespace-normal break-all">${escapeHtml(card.callout)}</p>
    </div>
    <ul class="flex flex-col gap-3 pl-8 pr-2 list-disc marker:text-[#7B5B52]">${bullets}</ul>
  </div>
  <div class="absolute bottom-12 right-16 text-[20px] text-[#8B7355]">www.snapdesign.app</div>
</div>`;
}

function renderCardTemplateB(card) {
  const headingClass = headingClassByLength(card.headline);
  const leadClass = leadClassByLength(card.lead);
  const bullets = card.bullets
    .map(
      (item, idx) => `<div class="rounded-[18px] border border-[#D4A574]/45 bg-[#D4A574]/16 px-5 py-4">
  <div class="text-[18px] tracking-[0.12em] text-[#8F756A] mb-1">POINT ${String(idx + 1).padStart(2, '0')}</div>
  <div class="text-[30px] leading-[1.42] text-[#3E2723] whitespace-normal break-all">${escapeHtml(item)}</div>
</div>`
    )
    .join('');

  return `<div class="relative w-[900px] h-[1198px] bg-[#FFFCF8] flex flex-col justify-start py-[84px] px-[76px] overflow-hidden shadow-lg border border-neutral-200 rounded-[26px]">
  <div class="absolute top-0 left-0 right-0 h-[14px] bg-[#6F4A42]"></div>
  <div class="flex items-center justify-between mt-2">
    <div class="inline-flex items-center gap-3">
      <div class="w-[72px] h-[72px] rounded-full bg-[#6F4A42] text-white text-[31px] font-bold flex items-center justify-center">${escapeHtml(card.stage)}</div>
      <div class="text-[22px] tracking-[0.12em] text-[#8F756A] uppercase">${escapeHtml(card.kicker)}</div>
    </div>
  </div>
  <div class="mt-8 flex flex-col gap-7">
    <h1 class="font-serif ${headingClass} leading-[1.08] text-[#2D1E19] whitespace-normal break-all">${escapeHtml(card.headline)}</h1>
    <p class="${leadClass} text-[#5A3E35] whitespace-normal break-all">${escapeHtml(card.lead)}</p>
    <div class="rounded-[24px] border border-[#A8B5A0]/45 bg-[#A8B5A0]/18 px-7 py-6">
      <div class="text-[21px] tracking-[0.12em] text-[#6F4A42] mb-2 uppercase">Core Message</div>
      <p class="text-[33px] leading-[1.45] font-semibold text-[#3E2723] whitespace-normal break-all">${escapeHtml(card.callout)}</p>
    </div>
    <div class="grid grid-cols-1 gap-3">${bullets}</div>
  </div>
  <div class="absolute bottom-12 right-16 text-[20px] text-[#8B7355]">www.snapdesign.app</div>
</div>`;
}

function renderCardsFromPlan(plan) {
  const cards = plan.cards.map((card, idx) => (idx % 2 === 0 ? renderCardTemplateA(card) : renderCardTemplateB(card)));
  return `<div class="w-full flex flex-col items-center gap-[20px] bg-[#F5F5F5] py-8">${cards.join('\n')}</div>`;
}

function buildRedNotePrompt(text, wantsImages = false, cardCount = 3, { strictMode = false } = {}) {
  const imageInstruction = wantsImages
    ? 'IMAGES: You may use image placeholders if necessary.'
    : 'IMAGES: Do not output any image placeholder/gray media block in text-only mode.';

  return `Create ${cardCount} Xiaohongshu cards from:
"${text}"

Output only Tailwind HTML.
Start with a wrapper div.
${imageInstruction}

Each card must use:
class="relative w-[900px] h-[1198px] bg-[#FFFCF8] ...".

Design requirements:
- strong typography hierarchy
- information-dense layout
- no giant blank region
- include section blocks and bullet points
- no emojis

${strictMode ? 'STRICT: ensure card count exact, avoid empty wrappers, keep rich visual blocks.' : ''}`;
}

function getHtmlSystemInstruction() {
  return 'You are an expert Tailwind designer. Output valid HTML only, no markdown.';
}

function sanitizeGeneratedHtml(rawHtml, { wantsImages = false } = {}) {
  let html = stripMarkdownFences(rawHtml);
  if (!wantsImages) {
    const patterns = [
      /<div[^>]*data-img-placeholder[^>]*>[\s\S]*?<\/div>/gi,
      /<div[^>]*class=["'][^"']*(?:bg-gray-(?:100|200|300)|bg-slate-(?:100|200|300)|bg-neutral-(?:100|200|300))[^"']*(?:aspect-\[[^"']+\]|h-\[\d+px\]|min-h-\[\d+px\])[^"']*["'][^>]*>(?:\s|&nbsp;|<br\s*\/?>|<!--[\s\S]*?-->)*<\/div>/gi,
    ];
    for (const pattern of patterns) {
      html = html.replace(pattern, '');
    }
  }
  html = html.replace(/\[\[LOADING_IMAGE_PLACEHOLDER\]\]/g, '');
  html = html.replace(/<div[^>]*>\s*<\/div>/g, '');
  return html.trim();
}

function countCardsInHtml(html) {
  const matches = String(html || '').match(/class=["'][^"']*w-\[900px\][^"']*h-\[1198px\][^"']*["']/gi);
  return matches ? matches.length : 0;
}

function analyzeHtmlQuality(html, { wantsImages = false, expectedCardCount = 0 } = {}) {
  const normalized = String(html || '');
  const plainTextLength = normalized.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().length;
  const cards = countCardsInHtml(normalized);
  const issues = [];

  const headingCount = (normalized.match(/text-\[(?:86|76|66|58)px\]|text-(?:6xl|5xl|4xl)/g) || []).length;
  const sectionBlocks = (normalized.match(/rounded-\[[0-9]+px\]|rounded-[0-9]xl|border/g) || []).length;
  const pointLikeLines = (normalized.match(/<li\b|POINT\s+\d{2}|第[一二三四五六七八九十]|\d+\./g) || []).length;
  const avgTextPerCard = cards > 0 ? Math.floor(plainTextLength / cards) : 0;

  if (cards <= 0) issues.push('no_card_container');
  if (expectedCardCount > 0 && cards !== expectedCardCount) issues.push(`card_count_mismatch_${cards}_of_${expectedCardCount}`);
  if (plainTextLength < 120) issues.push('text_too_short');
  if (cards > 0 && avgTextPerCard < 110) issues.push('low_information_density');
  if (headingCount < Math.max(1, cards)) issues.push('missing_heading_hierarchy');
  if (sectionBlocks < Math.max(3, cards * 3)) issues.push('too_few_visual_sections');
  if (pointLikeLines < Math.max(4, cards * 2)) issues.push('too_few_key_points');

  if (!wantsImages) {
    if (/data-img-placeholder/i.test(normalized)) issues.push('has_image_placeholder_attr');
    if (/bg-gray-(100|200|300)/i.test(normalized)) issues.push('has_gray_placeholder_block');
  }

  return { ok: issues.length === 0, issues, cards, plainTextLength, avgTextPerCard };
}

function buildFallbackCardsHtml(text, title, cardCount = 3) {
  const blocks = splitTextBlocks(text, cardCount).slice(0, Math.max(1, cardCount));
  const cards = blocks.map((block, idx) => {
    const stage = String(idx + 1).padStart(2, '0');
    return `<div class="relative w-[900px] h-[1198px] bg-[#FFFCF8] flex flex-col justify-start py-[84px] px-[76px] overflow-hidden shadow-lg border border-neutral-200 rounded-[26px]">
  <div class="absolute top-0 left-0 right-0 h-[14px] bg-[#6F4A42]"></div>
  <div class="flex items-center justify-between mt-2">
    <div class="w-[72px] h-[72px] rounded-full bg-[#6F4A42] text-white text-[31px] font-bold flex items-center justify-center">${stage}</div>
    <div class="text-[24px] tracking-[0.18em] uppercase text-[#8F756A]">${KICKERS[idx % KICKERS.length]}</div>
  </div>
  ${idx === 0 && title ? `<h1 class="mt-8 font-serif text-[72px] leading-[1.08] text-[#2D1E19]">${escapeHtml(clipText(title, 26))}</h1>` : ''}
  <p class="mt-8 text-[38px] leading-[1.5] text-[#5A3E35] whitespace-normal break-all">${escapeHtml(clipText(block, 160))}</p>
  <div class="mt-7 rounded-[24px] border border-[#E8B4A0]/45 bg-[#E8B4A0]/30 p-7 text-[32px] leading-[1.45] text-[#3E2723]">核心观点：把重复动作固化为可复用流程。</div>
  <ul class="mt-6 flex flex-col gap-3 pl-8 list-disc marker:text-[#7B5B52]">
    <li class="text-[31px] leading-[1.42] text-[#3E2723]">提炼关键信息</li>
    <li class="text-[31px] leading-[1.42] text-[#3E2723]">保持字号层级</li>
    <li class="text-[31px] leading-[1.42] text-[#3E2723]">增强版面饱满度</li>
  </ul>
  <div class="absolute bottom-12 right-16 text-[20px] text-[#8B7355]">www.snapdesign.app</div>
</div>`;
  });

  return `<div class="w-full flex flex-col items-center gap-[20px] bg-[#F5F5F5] py-8">${cards.join('\n')}</div>`;
}

async function generateWithLLM(userPrompt, systemInstruction) {
  if (!LLM_API_KEY) {
    throw new Error('LLM_API_KEY environment variable is required');
  }

  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${LLM_API_KEY}`,
  };
  if (LLM_BASE_URL.includes('openrouter.ai')) {
    headers['HTTP-Referer'] = 'https://snapdesign.app';
    headers['X-Title'] = 'SnapDesign RedNote';
  }

  const response = await fetch(`${LLM_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: LLM_HTML_MODEL,
      messages: [
        { role: 'system', content: systemInstruction },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.55,
      max_tokens: 4000,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`LLM API error: ${error}`);
  }

  const data = await response.json();
  const message = data?.choices?.[0]?.message;
  const content = message?.content || message?.reasoning || '';
  if (!content) {
    throw new Error(`Invalid LLM response format: ${JSON.stringify(data)}`);
  }
  return content;
}

function isRetryableLlmError(message) {
  const text = String(message || '').toLowerCase();
  return (
    text.includes('engine_overloaded_error') ||
    text.includes('overloaded') ||
    text.includes('rate limit') ||
    text.includes('429') ||
    text.includes('timeout') ||
    text.includes('temporar') ||
    text.includes('eai_again') ||
    text.includes('please try again later')
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function generateWithLLMRetry(userPrompt, systemInstruction, attempts = 3) {
  let lastError = null;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      return await generateWithLLM(userPrompt, systemInstruction);
    } catch (error) {
      lastError = error;
      const retryable = isRetryableLlmError(error?.message);
      if (!retryable || i === attempts) {
        throw error;
      }
      const waitMs = 600 * i;
      console.warn(`LLM temporary error (attempt ${i}/${attempts}), retrying in ${waitMs}ms...`);
      await sleep(waitMs);
    }
  }
  throw lastError || new Error('LLM request failed');
}

async function htmlToImage(html, outputPath) {
  const puppeteer = require('puppeteer');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const fullHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
  </style>
</head>
<body>${html}</body>
</html>`;

  try {
    const page = await browser.newPage();
    await page.setContent(fullHtml);
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const cardHtmls = await page.evaluate(() => {
      const cards = document.querySelectorAll('.w-\\[900px\\].h-\\[1198px\\], .w-\\[900px\\]');
      return Array.from(cards).map((card) => card.outerHTML);
    });
    if (!cardHtmls.length) {
      throw new Error('No card containers found for image rendering.');
    }

    for (let i = 0; i < cardHtmls.length; i += 1) {
      const singleCardHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 900px;
      height: 1198px;
      overflow: hidden;
      font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    }
  </style>
</head>
<body>${cardHtmls[i]}</body>
</html>`;

      await page.setViewport({ width: 900, height: 1198 });
      await page.setContent(singleCardHtml);
      await new Promise((resolve) => setTimeout(resolve, 1200));

      const cardPath = outputPath.replace('.png', `-${i + 1}.png`);
      await page.screenshot({
        path: cardPath,
        type: 'png',
        clip: { x: 0, y: 0, width: 900, height: 1198 },
        omitBackground: false,
      });
      console.log(`OK card ${i + 1}/${cardHtmls.length}: ${cardPath}`);
    }

    return cardHtmls.length;
  } finally {
    await browser.close();
  }
}

function detectUserIntent(text) {
  const source = String(text || '').toLowerCase();
  const imageKeywords = ['\u914d\u56fe', '\u63d2\u56fe', '\u56fe\u7247', '\u56fe\u50cf', '\u5c01\u9762\u56fe', 'illustration', 'image', 'with image'];
  const noImageKeywords = ['\u4e0d\u8981\u56fe', '\u65e0\u56fe', '\u4e0d\u9700\u8981\u56fe\u7247', '\u7eaf\u6587\u5b57', 'text only', 'without image', 'no image'];
  const wantsImages = imageKeywords.some((kw) => source.includes(kw.toLowerCase()));
  const rejectsImages = noImageKeywords.some((kw) => source.includes(kw.toLowerCase()));
  return { wantsImages, rejectsImages };
}

function printHelp() {
  console.log(`
SnapDesign RedNote v2

Usage:
  node generate-v2.js "your content" [options]

Options:
  --title "title"       Optional card title
  --output <dir>        Output directory (default: ./output-v2)
  --cards <n>           Card count (default: auto, 3-9)
  --with-images         Force image-placeholder mode

Env vars:
  LLM_API_KEY           Required
  LLM_BASE_URL          OpenAI-compatible base url, e.g. https://api.openai.com/v1
  LLM_MODEL             Model id, e.g. gpt-4o-mini / moonshot-v1-8k

Example:
  export LLM_API_KEY="your-key"
  export LLM_BASE_URL="https://api.openai.com/v1"
  export LLM_MODEL="gpt-4o-mini"
  node generate-v2.js "分享3个提升专注力的方法" --title "高效专注指南"
`);
}

async function main() {
  const args = process.argv.slice(2);
  if (!args.length || args[0] === '--help') {
    printHelp();
    return;
  }

  const content = args[0];
  let title = '';
  let outputDir = './output-v2';
  let cardCount = null;
  let forceImages = false;

  for (let i = 1; i < args.length; i += 1) {
    if (args[i] === '--title' && args[i + 1]) {
      title = args[i + 1];
      i += 1;
    } else if (args[i] === '--output' && args[i + 1]) {
      outputDir = args[i + 1];
      i += 1;
    } else if (args[i] === '--cards' && args[i + 1]) {
      cardCount = parseInt(args[i + 1], 10);
      i += 1;
    } else if (args[i] === '--with-images') {
      forceImages = true;
    }
  }

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  if (!LLM_API_KEY) {
    console.error('ERROR: missing LLM_API_KEY.');
    process.exit(1);
  }

  const fullText = title ? `${title}\n\n${content}` : content;
  const { wantsImages, rejectsImages } = detectUserIntent(fullText);
  const needImages = forceImages || (wantsImages && !rejectsImages);

  if (!cardCount) {
    const textLength = fullText.length;
    cardCount = Math.min(Math.max(Math.ceil(textLength / 150), 3), 9);
  }

  console.log(`Generating ${cardCount} cards with model: ${LLM_HTML_MODEL}`);
  console.log(`Image mode: ${needImages ? 'on' : 'off'}`);
  if (LLM_IMAGE_MODEL) {
    console.log(`Image model hint: ${LLM_IMAGE_MODEL}`);
  }

  try {
    let cleanHtml = '';
    let quality = { ok: false, issues: ['not_generated'] };

    try {
      const planPrompt = buildStructuredPlanPrompt(fullText, cardCount, title);
      const planRaw = await generateWithLLMRetry(planPrompt, getPlanSystemInstruction(), 3);
      const parsedPlan = parseJsonPayload(planRaw);
      const normalizedPlan = normalizeCardPlan(parsedPlan, fullText, title, cardCount);
      cleanHtml = renderCardsFromPlan(normalizedPlan);
      quality = analyzeHtmlQuality(cleanHtml, { wantsImages: needImages, expectedCardCount: cardCount });
      console.log(`Structured-template mode quality: ${quality.ok ? 'pass' : `fail (${quality.issues.join(', ')})`}`);
    } catch (planError) {
      console.warn(`Structured-template mode error: ${planError.message}`);
    }

    if (!quality.ok) {
      try {
        const systemInstruction = getHtmlSystemInstruction();
        const prompt = buildRedNotePrompt(fullText, needImages, cardCount, { strictMode: false });
        let html = await generateWithLLMRetry(prompt, systemInstruction, 3);
        cleanHtml = sanitizeGeneratedHtml(html, { wantsImages: needImages });
        quality = analyzeHtmlQuality(cleanHtml, { wantsImages: needImages, expectedCardCount: cardCount });

        if (!quality.ok) {
          console.warn(`Direct-HTML quality failed: ${quality.issues.join(', ')}`);
          const strictPrompt = buildRedNotePrompt(fullText, needImages, cardCount, { strictMode: true });
          html = await generateWithLLMRetry(strictPrompt, systemInstruction, 3);
          cleanHtml = sanitizeGeneratedHtml(html, { wantsImages: needImages });
          quality = analyzeHtmlQuality(cleanHtml, { wantsImages: needImages, expectedCardCount: cardCount });
        }
      } catch (htmlModeError) {
        console.warn(`Direct-HTML mode error: ${htmlModeError.message}`);
        quality = { ok: false, issues: ['direct_html_generation_failed'] };
      }
    }

    if (!quality.ok) {
      console.warn(`Strict mode failed: ${quality.issues.join(', ')}`);
      cleanHtml = buildFallbackCardsHtml(fullText, title, cardCount);
      quality = analyzeHtmlQuality(cleanHtml, { wantsImages: needImages, expectedCardCount: cardCount });
      console.warn(`Fallback applied (cards=${quality.cards}, avgTextPerCard=${quality.avgTextPerCard}).`);
    }

    const htmlPath = path.join(outputDir, 'cards.html');
    const fullHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      background: #F5F5F5;
    }
  </style>
</head>
<body>${cleanHtml}</body>
</html>`;
    fs.writeFileSync(htmlPath, fullHtml);
    console.log(`Saved HTML: ${htmlPath}`);

    const count = await htmlToImage(cleanHtml, path.join(outputDir, 'card.png'));
    console.log(`Done: generated ${count} cards to ${outputDir}`);
  } catch (error) {
    console.error(`ERROR: ${error.message}`);
    if (error.message.includes('API')) {
      console.error('Check LLM_API_KEY / LLM_BASE_URL / LLM_MODEL configuration.');
    }
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error(`ERROR: ${err.message}`);
    process.exit(1);
  });
}

module.exports = {
  buildStructuredPlanPrompt,
  normalizeCardPlan,
  renderCardsFromPlan,
  buildRedNotePrompt,
  sanitizeGeneratedHtml,
  analyzeHtmlQuality,
  detectUserIntent,
};
