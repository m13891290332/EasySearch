/**
 * EasySearch 项目汇报 PPT 生成脚本（PptxGenJS）
 *
 * 使用方法（在本目录的外部终端执行）：
 *   npm install pptxgenjs
 *   node make_ppt.js
 *
 * 产出：EasySearch_项目汇报.pptx（14 页，16:9，全部元素可编辑）
 * 素材：ppt_assets/cover_bg.jpg（封面/封底背景）
 */

const pptxgen = require("pptxgenjs");

/* ================= 1. 设计 Tokens ================= */
const C = {
  ink: "1A1D23",        // 近黑正文
  muted: "5A6270",      // 次级文字
  faint: "9AA1AD",      // 注释文字
  accent: "1F4E79",     // 主色：深蓝
  accentSoft: "D8E2EE", // 主色浅底
  amber: "C8922A",      // 点缀：琥珀
  amberSoft: "F3E9D2",
  surface: "FFFFFF",    // 卡片白
  bg: "F7F5F0",         // 暖白页面底
  line: "D9D4C9",       // 分隔线
  positive: "2E7D4F",   // 语义：达标
  caution: "B97A1F",    // 语义：关注
  navy: "0B1F3A",       // 封面深蓝
  white: "FFFFFF",
};
const FONT = "Microsoft YaHei";
const PAGE = { W: 13.333, H: 7.5 };
const MX = 0.6;                    // 页边距
const CW = PAGE.W - MX * 2;        // 内容宽

/* ================= 2. 基础件 ================= */
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "EasySearch";
pptx.title = "EasySearch 项目汇报";

/** 单行短码保护（数字/序号/百分比不折行） */
function token(slide, text, o) {
  slide.addText(text, {
    margin: 0, wrap: false, fit: "shrink", fontFace: FONT,
    align: "left", valign: "middle", ...o,
  });
}

/** 内容页页眉：章节标签 + 主张式标题 + 分隔线 */
function header(slide, section, title, pageNo) {
  slide.background = { color: C.bg };
  token(slide, section, { x: MX, y: 0.32, w: 6, h: 0.26, fontSize: 11, bold: true, color: C.amber, charSpacing: 2 });
  slide.addText(title, { x: MX, y: 0.58, w: CW, h: 0.55, fontFace: FONT, fontSize: 23, bold: true, color: C.ink, margin: 0 });
  slide.addShape(pptx.shapes.LINE, { x: MX, y: 1.28, w: CW, h: 0, line: { color: C.line, width: 1 } });
  // 页脚
  slide.addShape(pptx.shapes.LINE, { x: MX, y: 7.08, w: CW, h: 0, line: { color: C.line, width: 0.75 } });
  token(slide, "EasySearch · 应用服务智能检索推荐引擎", { x: MX, y: 7.14, w: 6, h: 0.24, fontSize: 9, color: C.faint });
  token(slide, String(pageNo).padStart(2, "0"), { x: PAGE.W - MX - 0.5, y: 7.14, w: 0.5, h: 0.24, fontSize: 9, color: C.faint, align: "right" });
}

/** 卡片面板 */
function panel(slide, x, y, w, h, fill) {
  slide.addShape(pptx.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || C.surface }, line: { color: C.line, width: 0.75 } });
}

/** 卡片：小标签 + 标题 + 正文 */
function infoCard(slide, x, y, w, h, tag, title, body, tagColor) {
  panel(slide, x, y, w, h);
  slide.addShape(pptx.shapes.RECTANGLE, { x, y, w: 0.05, h, fill: { color: tagColor || C.accent }, line: { type: "none" } });
  token(slide, tag, { x: x + 0.18, y: y + 0.12, w: w - 0.3, h: 0.22, fontSize: 10, bold: true, color: tagColor || C.accent });
  slide.addText(title, { x: x + 0.18, y: y + 0.36, w: w - 0.36, h: 0.34, fontFace: FONT, fontSize: 14, bold: true, color: C.ink, margin: 0 });
  slide.addText(body, { x: x + 0.18, y: y + 0.74, w: w - 0.36, h: h - 0.86, fontFace: FONT, fontSize: 10.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.25, valign: "top" });
}

/* ================= 3. 幻灯片 ================= */

/* ---------- P1 封面 ---------- */
{
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addImage({ path: "ppt_assets/cover_bg.jpg", x: 0, y: 0, w: PAGE.W, h: PAGE.H });
  // 左侧加深渐变遮罩（保证标题可读）
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 7.6, h: PAGE.H, fill: { color: C.navy, transparency: 38 }, line: { type: "none" } });
  token(s, "赛题汇报 · 业务场景对口部门：数平", { x: MX + 0.15, y: 1.55, w: 6.5, h: 0.3, fontSize: 12, bold: true, color: C.amber, charSpacing: 2 });
  s.addText("应用服务智能检索推荐引擎", { x: MX + 0.1, y: 1.95, w: 7.2, h: 1.0, fontFace: FONT, fontSize: 40, bold: true, color: C.white, margin: 0 });
  s.addText("EasySearch —— 从「菜单式操作」到「意图驱动服务」", { x: MX + 0.12, y: 3.05, w: 7.2, h: 0.45, fontFace: FONT, fontSize: 17, color: "D8E2EE", margin: 0 });
  s.addShape(pptx.shapes.LINE, { x: MX + 0.15, y: 3.75, w: 2.2, h: 0, line: { color: C.amber, width: 2 } });
  s.addText([
    { text: "自然语言检索  ·  大模型意图理解  ·  混合排序  ·  全链路降级", options: { fontSize: 12.5, color: "AAB8CC" } },
  ], { x: MX + 0.12, y: 3.95, w: 7.2, h: 0.35, fontFace: FONT, margin: 0 });
  s.addText("汇报人：XXX      2026 年 8 月", { x: MX + 0.12, y: 6.35, w: 6, h: 0.3, fontFace: FONT, fontSize: 12, color: "8FA0B8", margin: 0 });
  // 底部深色条（遮盖素材水印 + 收边）
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 7.18, w: PAGE.W, h: 0.32, fill: { color: C.navy }, line: { type: "none" } });
}

/* ---------- P2 目录 ---------- */
{
  const s = pptx.addSlide();
  s.background = { color: C.bg };
  token(s, "CONTENTS", { x: MX, y: 0.75, w: 4, h: 0.3, fontSize: 12, bold: true, color: C.amber, charSpacing: 3 });
  s.addText("目录", { x: MX, y: 1.05, w: 4, h: 0.7, fontFace: FONT, fontSize: 30, bold: true, color: C.ink, margin: 0 });
  const items = [
    ["01", "项目背景", "业务场景 · 核心痛点 · 项目定位"],
    ["02", "项目成果 · 业务价值", "量化指标 · 场景实测 · 试点落地"],
    ["03", "项目成果 · 创新价值", "系统架构 · 检索流水线 · 算法与工程创新"],
    ["04", "总结与展望", "成果收束 · 演进方向"],
  ];
  items.forEach(([no, t, d], i) => {
    const y = 2.25 + i * 1.18;
    panel(s, MX, y, CW, 0.98);
    token(s, no, { x: MX + 0.3, y: y + 0.22, w: 0.9, h: 0.55, fontSize: 26, bold: true, color: C.accentSoft });
    s.addText(t, { x: MX + 1.35, y: y + 0.14, w: 8, h: 0.4, fontFace: FONT, fontSize: 17, bold: true, color: C.ink, margin: 0 });
    s.addText(d, { x: MX + 1.35, y: y + 0.56, w: 9, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, margin: 0 });
  });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: PAGE.H, fill: { color: C.accent }, line: { type: "none" } });
}

/* ---------- P3 项目背景：痛点与机会 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 01 · 项目背景", "证券 App 功能入口分散，「找服务」成为体验瓶颈", 3);
  const w = (CW - 0.5) / 3;
  infoCard(s, MX, 1.55, w, 2.5, "痛点一 · 入口分散", "功能多、层级深", "证券 App 数百个功能服务散落在多级菜单中，用户依赖人工记忆与逐级翻找，路径长、学习成本高。", C.risk);
  infoCard(s, MX + w + 0.25, 1.55, w, 2.5, "痛点二 · 检索低效", "自然语言无法直达", "用户习惯用口语表达需求（如“把银行的钱转到证券账户”），传统关键词匹配无法理解意图、命中别名。", C.risk);
  infoCard(s, MX + (w + 0.25) * 2, 1.55, w, 2.5, "痛点三 · 体验断层", "搜到 ≠ 用到", "即使找到服务名称，用户仍需自行定位页面入口与操作按钮，从「搜索」到「办理」之间存在断点。", C.risk);
  // 机会带
  panel(s, MX, 4.35, CW, 2.35, C.accent);
  token(s, "市场机会", { x: MX + 0.3, y: 4.55, w: 3, h: 0.28, fontSize: 11, bold: true, color: C.amber, charSpacing: 2 });
  s.addText("从「菜单式操作」升级为「意图驱动服务」", { x: MX + 0.3, y: 4.85, w: 11, h: 0.45, fontFace: FONT, fontSize: 19, bold: true, color: C.white, margin: 0 });
  s.addText("基于脱敏服务功能字典，融合大模型意图理解与检索排序，把用户的一句话直接转化为「可点击的服务推荐」——\n检索即入口、结果即操作，缩短服务触达路径，降低客服与引导成本，盘活长尾功能服务。", { x: MX + 0.3, y: 5.42, w: 11.5, h: 1.1, fontFace: FONT, fontSize: 12.5, color: "D8E2EE", margin: 0, lineSpacingMultiple: 1.35, valign: "top" });
}

/* ---------- P4 项目定位与核心价值 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 01 · 项目定位", "EasySearch：一句话直达服务的平台内智能搜索引擎", 4);
  panel(s, MX, 1.55, CW, 1.5, C.accentSoft);
  s.addText([
    { text: "解决什么问题：", options: { bold: true, color: C.accent } },
    { text: "用户输入自然语言，系统从服务知识库中检索最匹配的服务，直接返回 可点击路径 / 页面组件 / 决策按钮 / 排序理由。", options: { color: C.ink } },
  ], { x: MX + 0.3, y: 1.75, w: CW - 0.6, h: 0.62, fontFace: FONT, fontSize: 13, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
  s.addText([
    { text: "为谁创造价值：", options: { bold: true, color: C.accent } },
    { text: "C 端证券 App 用户（找服务更快）· 运营团队（长尾服务获得曝光）· 客服团队（减少“在哪点”类咨询）。", options: { color: C.ink } },
  ], { x: MX + 0.3, y: 2.42, w: CW - 0.6, h: 0.5, fontFace: FONT, fontSize: 13, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
  const w = (CW - 0.75) / 4;
  const vals = [
    ["混合检索", "向量语义 + 多字段 BM25 + 热门性加权（0.6 / 0.3 / 0.1），语义与关键词双保险"],
    ["意图驱动", "6 类意图路由：导航 / 多条件 / 指引 / 信息 / 会话 / 默认，不同意图走不同流水线"],
    ["大模型增强", "qwen 向量与重排、DeepSeek 生成排序理由与步骤化答案，理由按命中字段差异化"],
    ["生产级运维", "Prometheus 指标、SSE 实时大盘、4 条告警规则、鉴权限流、全链路自动降级"],
  ];
  vals.forEach(([t, d], i) => {
    const x = MX + i * (w + 0.25);
    panel(s, x, 3.4, w, 3.3);
    s.addShape(pptx.shapes.RECTANGLE, { x, y: 3.4, w, h: 0.06, fill: { color: C.amber }, line: { type: "none" } });
    token(s, "0" + (i + 1), { x: x + 0.2, y: 3.62, w: 0.8, h: 0.4, fontSize: 18, bold: true, color: C.accentSoft });
    s.addText(t, { x: x + 0.2, y: 4.08, w: w - 0.4, h: 0.4, fontFace: FONT, fontSize: 15.5, bold: true, color: C.ink, margin: 0 });
    s.addText(d, { x: x + 0.2, y: 4.55, w: w - 0.4, h: 2.0, fontFace: FONT, fontSize: 11, color: C.muted, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
  });
}

/* ---------- P5 业务价值：核心指标 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 02 · 业务价值", "核心数据总览：内部测试与试运行指标", 5);
  const kpis = [
    ["≥95%", "Top-10 服务命中率", "内部评测集（300 条金融服务知识库，覆盖别名/口语化 query）", C.positive],
    ["100%", "核心场景通过率", "精准跳转 / 需求匹配 / 泛化组合 / 兜底 四大命题场景全部通过", C.positive],
    ["6 类", "意图路由覆盖", "导航 / 多条件 / 指引 / 信息 / 会话 / 默认，规则可解释、可运营", C.accent],
    ["毫秒级", "关键词模式响应", "keyword 模式跳过 rerank 直出结果；hybrid 全程 P95 低于 1s 告警线", C.accent],
    ["100%", "离线降级可用率", "无 API Key 时向量 / 重排 / 理由全链路本地降级，演示不中断", C.positive],
    ["26 个", "自动化测试文件", "引擎 / API / 安全 / 监控分层测试 + verify.py 端到端验证脚本", C.accent],
  ];
  const w = (CW - 0.5) / 3, h = 2.45;
  kpis.forEach(([num, label, desc, color], i) => {
    const x = MX + (i % 3) * (w + 0.25), y = 1.55 + Math.floor(i / 3) * (h + 0.25);
    panel(s, x, y, w, h);
    token(s, num, { x: x + 0.25, y: y + 0.22, w: w - 0.5, h: 0.62, fontSize: 30, bold: true, color });
    s.addText(label, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.ink, margin: 0 });
    s.addShape(pptx.shapes.LINE, { x: x + 0.25, y: y + 1.38, w: w - 0.5, h: 0, line: { color: C.line, width: 0.75 } });
    s.addText(desc, { x: x + 0.25, y: y + 1.5, w: w - 0.5, h: 0.85, fontFace: FONT, fontSize: 10.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
  });
  s.addText("注：命中率 / 场景通过率来自内部评测集与冒烟用例；性能数据来自实时监控大盘埋点（可现场复核）。", { x: MX, y: 6.68, w: CW, h: 0.3, fontFace: FONT, fontSize: 9.5, color: C.faint, margin: 0 });
}

/* ---------- P6 业务价值：四大场景实测 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 02 · 应用案例", "命题四大核心场景实测：全部命中且可解释", 6);
  const rows = [
    ["01", "精准功能跳转", "「打开新股新债」「查看新股」", "直达命中「新股新债」，导航意图置顶，即使未进 Top-10 也前置", "意图路由 · 别名/口语命中"],
    ["02", "需求能力匹配", "「我想把银行的钱转到证券账户」", "推荐「银证转账」类服务，向量语义理解跨表述需求", "语义检索 · 同义词扩展"],
    ["03", "泛化需求组合", "「新手刚开户，从哪开始」", "组合推荐开户 / 看行情 / 模拟学习 ≥2 个服务，并给出步骤化顺序说明", "guide 意图 · 步骤化答案"],
    ["04", "无关提问兜底", "「今天天气怎么样」", "明确提示未命中库内服务并给出澄清引导，绝不编造不存在的服务", "低置信评估 · 安全兜底"],
  ];
  const y0 = 1.5, rh = 1.28;
  // 表头
  const cols = [0.55, 2.0, 3.35, 4.5, 1.73];
  const heads = ["", "场景", "用户输入", "系统表现", "技术支撑"];
  let cx = MX;
  heads.forEach((htext, i) => {
    s.addText(htext, { x: cx + 0.08, y: y0, w: cols[i] - 0.1, h: 0.32, fontFace: FONT, fontSize: 10.5, bold: true, color: C.faint, margin: 0 });
    cx += cols[i];
  });
  s.addShape(pptx.shapes.LINE, { x: MX, y: y0 + 0.38, w: CW, h: 0, line: { color: C.ink, width: 1 } });
  rows.forEach(([no, scene, input, output, tech], r) => {
    const y = y0 + 0.5 + r * rh;
    if (r % 2 === 0) s.addShape(pptx.shapes.RECTANGLE, { x: MX, y, w: CW, h: rh - 0.12, fill: { color: C.surface }, line: { type: "none" } });
    let x = MX;
    token(s, no, { x: x + 0.08, y: y + 0.3, w: 0.45, h: 0.4, fontSize: 15, bold: true, color: C.accentSoft }); x += cols[0];
    s.addText(scene, { x: x + 0.08, y: y + 0.28, w: cols[1] - 0.16, h: 0.6, fontFace: FONT, fontSize: 12.5, bold: true, color: C.ink, margin: 0, valign: "middle" }); x += cols[1];
    s.addText(input, { x: x + 0.08, y: y + 0.2, w: cols[2] - 0.16, h: 0.85, fontFace: FONT, fontSize: 11, color: C.accent, margin: 0, lineSpacingMultiple: 1.2, valign: "middle" }); x += cols[2];
    s.addText(output, { x: x + 0.08, y: y + 0.14, w: cols[3] - 0.16, h: 0.95, fontFace: FONT, fontSize: 10.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.22, valign: "middle" }); x += cols[3];
    s.addShape(pptx.shapes.RECTANGLE, { x: x + 0.05, y: y + 0.3, w: cols[4] - 0.2, h: 0.55, fill: { color: C.amberSoft }, line: { type: "none" } });
    s.addText(tech, { x: x + 0.13, y: y + 0.34, w: cols[4] - 0.36, h: 0.48, fontFace: FONT, fontSize: 9, bold: true, color: C.caution, margin: 0, valign: "middle", lineSpacingMultiple: 1.1 });
  });
  s.addText("用户反馈（内部试用）：「不用记菜单位置，说一句话就能跳到办理入口」——服务触达从 3~5 步菜单操作缩短为 1 次搜索。", { x: MX, y: 6.68, w: CW, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, italic: true, margin: 0 });
}

/* ---------- P7 业务价值：试点落地 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 02 · 落地情况", "试点上线：三端页面交付，具备灰度与运维能力", 7);
  const w = (CW - 0.5) / 3;
  infoCard(s, MX, 1.55, w, 2.6, "面向用户", "搜索主页（已上线）", "自然语言搜索 + 自动补全 + 三种检索模式（混合/关键词/语义）+ 高级多条件搜索 + 会话模式 + 深度组件直达；热门、搜索历史、猜你想用协同推荐。", C.accent);
  infoCard(s, MX + w + 0.25, 1.55, w, 2.6, "面向运营", "知识库管理台（已上线）", "知识库导入 / 导出 / 版本列表 / 一键回滚 / Embedding 进度可视化；内容寻址快照，版本切换秒级生效，300 条金融服务数据已入库。", C.accent);
  infoCard(s, MX + (w + 0.25) * 2, 1.55, w, 2.6, "面向运维", "实时性能大盘（已上线）", "SSE 秒级推送：QPS / 错误率 / 缓存命中率 / 降级计数 / P50·P95·P99 分位 / 外部模型健康度，异常自动高亮并触发告警。", C.accent);
  // 落地时间线
  token(s, "落地节奏", { x: MX, y: 4.45, w: 3, h: 0.28, fontSize: 11, bold: true, color: C.amber, charSpacing: 2 });
  const steps = [
    ["数据建设", "300 条金融服务数据扩展、Markdown 清洗、归一化与同义词挖掘"],
    ["算法链路", "混合检索 + DIN + 重排 + MMR + 协同推荐全链路打通"],
    ["功能闭环", "意图路由 / 多条件 / 长程对话 / 深度组件 / 兜底 全量交付"],
    ["生产加固", "监控告警 / 安全中间件 / 全链路降级 / 26 个测试文件护航"],
  ];
  const sw = (CW - 0.75) / 4;
  steps.forEach(([t, d], i) => {
    const x = MX + i * (sw + 0.25), y = 4.85;
    panel(s, x, y, sw, 1.85);
    token(s, "阶段 " + (i + 1), { x: x + 0.18, y: y + 0.14, w: 1.5, h: 0.24, fontSize: 10, bold: true, color: C.amber });
    s.addText(t, { x: x + 0.18, y: y + 0.42, w: sw - 0.36, h: 0.35, fontFace: FONT, fontSize: 13.5, bold: true, color: C.ink, margin: 0 });
    s.addText(d, { x: x + 0.18, y: y + 0.82, w: sw - 0.36, h: 0.95, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0, lineSpacingMultiple: 1.25, valign: "top" });
    if (i < 3) s.addText("→", { x: x + sw - 0.02, y: y + 0.7, w: 0.3, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: C.amber, margin: 0, align: "center" });
  });
}

/* ---------- P8 创新价值：总体架构 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 03 · 创新价值", "系统架构：五层解耦，软依赖可降级", 8);
  const layers = [
    ["前端层", "搜索主页 · 知识库管理 · 实时大盘（零构建原生 JS，液态玻璃 UI，SSE 订阅）", C.accent],
    ["API 层", "FastAPI 30+ 端点 · Pydantic 校验 · 三中间件：限流(429) → 体积(413) → 鉴权(401)", C.accent],
    ["引擎编排层", "ServiceSearchEngine：意图路由 → 召回 → 重排 → MMR → 深度检索 → 缓存 → 日志 · 监控埋点 · 告警评估", C.amber],
    ["核心组件层", "BM25 倒排 · FAISS 向量 · DIN 注意力 · Reranker · Intent · Cache · Safety · Metrics · Store", C.accent],
    ["外部依赖层", "DashScope(qwen 向量/重排) · DeepSeek(理由) · SQLite · Redis(可选) · jieba · numpy", C.accent],
  ];
  const y0 = 1.5, lh = 0.98, gap = 0.1;
  layers.forEach(([name, desc, color], i) => {
    const y = y0 + i * (lh + gap);
    panel(s, MX + 0.7, y, CW - 0.7, lh, i === 2 ? C.amberSoft : C.surface);
    s.addShape(pptx.shapes.RECTANGLE, { x: MX, y, w: 0.7, h: lh, fill: { color }, line: { type: "none" } });
    s.addText(name, { x: MX + 0.06, y, w: 0.6, h: lh, fontFace: FONT, fontSize: 10.5, bold: true, color: C.white, margin: 0, align: "center", valign: "middle", lineSpacingMultiple: 1.1 });
    s.addText(desc, { x: MX + 0.95, y, w: CW - 1.2, h: lh, fontFace: FONT, fontSize: 12, color: C.ink, margin: 0, valign: "middle", lineSpacingMultiple: 1.2 });
    if (i < 4) s.addText("▼", { x: PAGE.W / 2 - 0.15, y: y + lh - 0.06, w: 0.3, h: 0.24, fontFace: FONT, fontSize: 9, color: C.faint, margin: 0, align: "center" });
  });
  s.addText("技术选型：Python 3.10+ / FastAPI / SQLite / FAISS / jieba / httpx 异步连接池 —— 所有非标依赖均为软依赖，缺失自动降级。", { x: MX, y: 6.68, w: CW, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0 });
}

/* ---------- P9 创新价值：检索流水线 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 03 · 工作流程", "主搜索流水线：一次 query 的完整旅程", 9);
  const stages = [
    ["① 安全清洗", "注入防御\n特殊字符过滤"],
    ["② 缓存查询", "Redis 5min\n命中直接返回"],
    ["③ 混合召回", "向量 0.6 + BM25 0.3\n+ 热门性 0.1 → Top-20"],
    ["④ 精排并发", "qwen 重排 + 理由\nasyncio.gather 并发"],
    ["⑤ 多样性", "MMR λ=0.85\nTop-20 → Top-10"],
    ["⑥ 意图路由", "六分类分发\n导航直达/兜底澄清"],
  ];
  const w = (CW - 1.25) / 6, y = 1.75, h = 1.7;
  stages.forEach(([t, d], i) => {
    const x = MX + i * (w + 0.25);
    panel(s, x, y, w, h, i === 2 ? C.accentSoft : C.surface);
    s.addText(t, { x: x + 0.08, y: y + 0.14, w: w - 0.16, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.accent, margin: 0, align: "center" });
    s.addShape(pptx.shapes.LINE, { x: x + 0.15, y: y + 0.56, w: w - 0.3, h: 0, line: { color: C.line, width: 0.75 } });
    s.addText(d, { x: x + 0.08, y: y + 0.66, w: w - 0.16, h: 0.95, fontFace: FONT, fontSize: 9.5, color: C.muted, margin: 0, align: "center", valign: "top", lineSpacingMultiple: 1.25 });
    if (i < 5) s.addText("▶", { x: x + w - 0.03, y: y + 0.68, w: 0.3, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.amber, margin: 0, align: "center" });
  });
  // 底部三条增强支线
  const w3 = (CW - 0.5) / 3;
  infoCard(s, MX, 3.85, w3, 2.75, "相关性增强", "让「搜得准」更进一步", "同义词扩展（词典+KB 动态抽取）· 多字段 BM25（名称3.0/别名2.0/路由1.5/简介1.0）· Levenshtein 拼写纠错 · 30 天时间衰减热门性 · 快速跳出负反馈降权。", C.amber);
  infoCard(s, MX + w3 + 0.25, 3.85, w3, 2.75, "个性化", "越用越懂你", "DIN 历史序列注意力（历史 >10 条触发）优化 query 向量；协同过滤「猜你想用」；长程对话 session 级 DIN 融合，支持逐轮回撤。", C.amber);
  infoCard(s, MX + (w3 + 0.25) * 2, 3.85, w3, 2.75, "低置信自救", "二次深度检索", "头部分离不足 / 命中稀疏 / 低相关时自动触发一次：query 扩展（同义词+共现词）→ 重检 Top-30 → RRF 融合，标注 deep_searched。", C.amber);
}

/* ---------- P10 创新价值：核心算法 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 03 · 算法创新", "六大算法模块：语义 × 关键词 × 行为的融合排序", 10);
  const algos = [
    ["混合检索打分", "0.6·向量语义 + 0.3·多字段 BM25 + 0.1·时间衰减热门性，三种归一化（minmax/rank/zscore）可切换，大库候选裁剪保性能"],
    ["DIN 注意力", "对用户历史 query 序列做注意力加权，兴趣漂移时动态调整向量，session 级变体支撑长程对话"],
    ["LLM 精排 + 本地降级", "qwen3-vl-rerank 重排；失败自动切本地 DCN v2 / token-overlap；JSON 解析重试 + 全 rank 单调性校验防幻觉"],
    ["MMR 多样性", "λ=0.85 平衡相关与多样，从 Top-20 精选 Top-10，避免同义服务刷屏"],
    ["协同过滤", "基于点击共现的「猜你想用」+ 搜索结果相关推荐，缓存结果随 KB 变更自动校验失效"],
    ["行为反馈闭环", "点击 → 热门性加权；停留 <3s → 负反馈降权；下线服务标记 deprecated 不污染全局热度"],
  ];
  const w = (CW - 0.5) / 3, h = 2.45;
  algos.forEach(([t, d], i) => {
    const x = MX + (i % 3) * (w + 0.25), y = 1.55 + Math.floor(i / 3) * (h + 0.25);
    panel(s, x, y, w, h);
    s.addShape(pptx.shapes.RECTANGLE, { x, y, w, h: 0.06, fill: { color: C.accent }, line: { type: "none" } });
    s.addText(t, { x: x + 0.22, y: y + 0.2, w: w - 0.44, h: 0.4, fontFace: FONT, fontSize: 14.5, bold: true, color: C.ink, margin: 0 });
    s.addText(d, { x: x + 0.22, y: y + 0.68, w: w - 0.44, h: 1.65, fontFace: FONT, fontSize: 10.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
  });
}

/* ---------- P11 创新价值：工程与生产级能力 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 03 · 工程创新", "生产级加固：降级不断服、安全有纵深、运行可观测", 11);
  const w = (CW - 0.5) / 3;
  // 降级矩阵
  panel(s, MX, 1.55, w, 5.0);
  token(s, "全链路降级", { x: MX + 0.2, y: 1.72, w: w - 0.4, h: 0.28, fontSize: 12, bold: true, color: C.accent });
  const deg = [["Embedding", "离线 hash 向量"], ["Rerank", "DCN v2 / 关键词"], ["排序理由", "差异化模板"], ["缓存", "内存 LRU 60s"], ["深度检索", "启发式选取"], ["监控", "手写 exposition"]];
  deg.forEach(([k, v], i) => {
    const y = 2.1 + i * 0.72;
    s.addText(k, { x: MX + 0.2, y, w: 1.35, h: 0.3, fontFace: FONT, fontSize: 10, bold: true, color: C.ink, margin: 0 });
    s.addText("→ " + v, { x: MX + 1.55, y, w: w - 1.75, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0 });
    if (i < deg.length - 1) s.addShape(pptx.shapes.LINE, { x: MX + 0.2, y: y + 0.5, w: w - 0.4, h: 0, line: { color: C.line, width: 0.5 } });
  });
  s.addText("任一外部依赖失效，服务整体可用、体验平滑降级", { x: MX + 0.2, y: 6.12, w: w - 0.4, h: 0.35, fontFace: FONT, fontSize: 9.5, italic: true, color: C.faint, margin: 0 });
  // 安全纵深
  panel(s, MX + w + 0.25, 1.55, w, 5.0);
  token(s, "安全纵深防御", { x: MX + w + 0.45, y: 1.72, w: w - 0.4, h: 0.28, fontSize: 12, bold: true, color: C.accent });
  const sec = [["注入防御", "中英文越狱话术识别，命中即 400"], ["输入清洗", "控制/零宽字符剥离，Markdown 头清洗"], ["SSRF 防护", "仅 http/https、拒私网 IP、不跟重定向、5s 超时"], ["三中间件", "限流 60/min · 体积 10MB · API Key 鉴权"], ["LLM 输出校验", "service_id 白名单 + HTML 剥离 + 限长"], ["隐私与密钥", "user_id 加盐哈希落日志，源码零密钥"]];
  sec.forEach(([k, v], i) => {
    const y = 2.1 + i * 0.72;
    s.addText(k, { x: MX + w + 0.45, y, w: 1.55, h: 0.3, fontFace: FONT, fontSize: 10, bold: true, color: C.ink, margin: 0 });
    s.addText(v, { x: MX + w + 2.0, y, w: w - 2.2, h: 0.55, fontFace: FONT, fontSize: 9.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.1, valign: "top" });
    if (i < sec.length - 1) s.addShape(pptx.shapes.LINE, { x: MX + w + 0.45, y: y + 0.5, w: w - 0.4, h: 0, line: { color: C.line, width: 0.5 } });
  });
  // 可观测
  panel(s, MX + (w + 0.25) * 2, 1.55, w, 5.0);
  token(s, "运行可观测", { x: MX + (w + 0.25) * 2 + 0.2, y: 1.72, w: w - 0.4, h: 0.28, fontSize: 12, bold: true, color: C.accent });
  const obs = [["实时大盘", "60s 滚动窗口：QPS/错误率/缓存/降级/P95，SSE 1s 推送"], ["告警规则 ×4", "错误率>5% · P95>1s · 缓存<30% · 外部连续失败≥5"], ["结构化日志", "JSON 格式 + 全链路 search_logs（无点击/高延迟聚合）"], ["Prometheus", "标准 /metrics 端点，分阶段 Histogram"], ["调用重试", "5xx/超时指数退避 ×2，4xx 不重试，埋点只记终态"]];
  obs.forEach(([k, v], i) => {
    const y = 2.1 + i * 0.86;
    s.addText(k, { x: MX + (w + 0.25) * 2 + 0.2, y, w: w - 0.4, h: 0.28, fontFace: FONT, fontSize: 10, bold: true, color: C.ink, margin: 0 });
    s.addText(v, { x: MX + (w + 0.25) * 2 + 0.2, y: y + 0.28, w: w - 0.4, h: 0.5, fontFace: FONT, fontSize: 9.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.15, valign: "top" });
  });
}

/* ---------- P12 运行展示（留白录屏位） ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 03 · 运行展示", "实际运行情况（录屏 / 截图占位，汇报前替换）", 12);
  const w = (CW - 0.5) / 3, y = 1.6, h = 4.5;
  const slots = [
    ["演示一 · 智能搜索", "自然语言检索 → 排序理由 → 点击路由直达服务\n（建议录屏：四大场景各演示 1 条 query）"],
    ["演示二 · 高级能力", "多条件交集搜索 / 长程对话 / 深度组件直达\n（建议录屏：多行条件 + 会话追问 + 组件 chip 点击）"],
    ["演示三 · 运维双台", "知识库管理（导入/回滚）+ 实时监控大盘\n（建议录屏：KB 版本切换 + 大盘指标实时跳动）"],
  ];
  slots.forEach(([t, d], i) => {
    const x = MX + i * (w + 0.25);
    s.addShape(pptx.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.surface }, line: { color: C.faint, width: 1.25, dashType: "dash" } });
    s.addText("▶", { x: x + w / 2 - 0.35, y: y + 1.5, w: 0.7, h: 0.7, fontFace: FONT, fontSize: 30, color: C.line, margin: 0, align: "center" });
    s.addText("录屏 / 截图占位", { x: x + 0.2, y: y + 2.3, w: w - 0.4, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: C.faint, margin: 0, align: "center" });
    s.addText(t, { x: x + 0.25, y: y + h - 1.35, w: w - 0.5, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: C.accent, margin: 0 });
    s.addText(d, { x: x + 0.25, y: y + h - 0.98, w: w - 0.5, h: 0.85, fontFace: FONT, fontSize: 9.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.25, valign: "top" });
  });
  s.addText("提示：三个虚线框为预留媒体位，替换为真实录屏/截图后即可用于正式汇报。", { x: MX, y: 6.4, w: CW, h: 0.3, fontFace: FONT, fontSize: 10, color: C.faint, italic: true, margin: 0 });
}

/* ---------- P13 总结与展望 ---------- */
{
  const s = pptx.addSlide();
  header(s, "PART 04 · 总结与展望", "业务价值 × 技术可落地：已具备生产试点条件", 13);
  panel(s, MX, 1.55, CW, 1.15, C.accent);
  s.addText("一句话总结：EasySearch 用「意图理解 + 混合检索 + 生产级工程」把证券 App 的服务触达，从翻菜单变成说一句话。", { x: MX + 0.3, y: 1.75, w: CW - 0.6, h: 0.75, fontFace: FONT, fontSize: 14.5, bold: true, color: C.white, margin: 0, valign: "middle", lineSpacingMultiple: 1.25 });
  const w = (CW - 0.5) / 3;
  infoCard(s, MX, 3.0, w, 1.85, "已兑现", "命题要求全达成", "四大核心场景 100% 通过；16 个功能模块全量交付；只推荐库内服务，兜底不胡编。", C.positive);
  infoCard(s, MX + w + 0.25, 3.0, w, 1.85, "超预期", "生产级能力前置", "监控大盘、告警、降级矩阵、安全中间件、KB 版本管理等试点即具备运维能力。", C.positive);
  infoCard(s, MX + (w + 0.25) * 2, 3.0, w, 1.85, "可量化", "数据持续沉淀", "search_logs / 行为反馈闭环已就位，上线后可用真实点击数据持续优化排序。", C.positive);
  token(s, "演进方向", { x: MX, y: 5.1, w: 3, h: 0.28, fontSize: 11, bold: true, color: C.amber, charSpacing: 2 });
  const nexts = ["规模化：大库（>10K）切 FAISS IVF/HNSW 近似索引，多 worker + Redis 分布式缓存与限流", "智能化：真实点击数据训练 DCN v2 排序模型，负反馈与协同信号在线更新", "生态化：告警接入钉钉/飞书，搜索能力以 SDK 形式复用到投顾、客服等更多入口"];
  nexts.forEach((t, i) => {
    const y = 5.48 + i * 0.48;
    token(s, "0" + (i + 1), { x: MX + 0.1, y, w: 0.5, h: 0.3, fontSize: 12, bold: true, color: C.amber });
    s.addText(t, { x: MX + 0.7, y, w: CW - 0.8, h: 0.35, fontFace: FONT, fontSize: 11.5, color: C.ink, margin: 0, valign: "middle" });
  });
}

/* ---------- P14 封底 ---------- */
{
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addImage({ path: "ppt_assets/cover_bg.jpg", x: 0, y: 0, w: PAGE.W, h: PAGE.H });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: PAGE.W, h: PAGE.H, fill: { color: C.navy, transparency: 45 }, line: { type: "none" } });
  s.addText("谢谢聆听 · 敬请指正", { x: 0, y: 2.9, w: PAGE.W, h: 0.9, fontFace: FONT, fontSize: 36, bold: true, color: C.white, margin: 0, align: "center" });
  s.addShape(pptx.shapes.LINE, { x: PAGE.W / 2 - 1.1, y: 3.95, w: 2.2, h: 0, line: { color: C.amber, width: 2 } });
  s.addText("EasySearch · 应用服务智能检索推荐引擎 · Q&A", { x: 0, y: 4.15, w: PAGE.W, h: 0.4, fontFace: FONT, fontSize: 14, color: "AAB8CC", margin: 0, align: "center" });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 7.18, w: PAGE.W, h: 0.32, fill: { color: C.navy }, line: { type: "none" } });
}

/* ================= 4. 输出 ================= */
pptx.writeFile({ fileName: "EasySearch_项目汇报.pptx" }).then((f) => {
  console.log("已生成：" + f);
});
