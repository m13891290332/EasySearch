// EasySearch 主题与设置：齿轮面板 / 主题切换（亮色随机渐变 · 暗色 WebGL 着色器 · 自定义背景图）
// / 光标跟随描边点亮 / 默认检索模式持久化 / 账号切换
// 尊重系统「减弱动态效果」偏好：关闭扫光与着色器 RAF，走静态兜底。

(function () {
  "use strict";

  var LS_USER = "easysearch_user";
  var LS_MODE = "easysearch_retrieval_mode";
  var LS_THEME = "easysearch_theme";     // light | dark | custom
  var LS_BG = "easysearch_bg";           // URL 或 dataURL

  var REDUCED =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(id) { return document.getElementById(id); }

  // ---------- 光标跟随描边点亮 ----------
  // 文档级 mousemove 委托：最近 .spotlight 祖先更新 --mx/--my（相对元素 px）
  document.addEventListener("mousemove", function (e) {
    var el = e.target.closest && e.target.closest(".spotlight");
    if (!el) return;
    var r = el.getBoundingClientRect();
    el.style.setProperty("--mx", e.clientX - r.left + "px");
    el.style.setProperty("--my", e.clientY - r.top + "px");
  });
  // 离开元素回中心（节流：仅对静态 spotlight 绑定）
  function resetSpotlight(el) {
    el.style.setProperty("--mx", "50%");
    el.style.setProperty("--my", "50%");
  }
  function bindSpotlightLeave() {
    document.querySelectorAll(".spotlight").forEach(function (el) {
      el.addEventListener("mouseleave", function () { resetSpotlight(el); });
    });
  }

  // ---------- 亮色随机渐变 ----------
  function randHsl() {
    var h = Math.floor(Math.random() * 360);
    var s = 60 + Math.floor(Math.random() * 20);
    var l = 78 + Math.floor(Math.random() * 12);
    return "hsl(" + h + "," + s + "%," + l + "%)";
  }
  function randomGradient() {
    var a = randHsl(), b = randHsl(), c = randHsl();
    var ang = 120 + Math.floor(Math.random() * 120);
    return "linear-gradient(" + ang + "deg, " + a + ", " + b + ", " + c + ")";
  }

  // ---------- 暗色动态着色器（WebGL plasma，reduced-motion 走静态） ----------
  var shader = { gl: null, prog: null, uTime: null, uRes: null, raf: 0, t: 0, canvas: null };
  var VERT = "attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}";
  var FRAG = [
    "precision highp float;",
    "uniform vec2 u_res;",
    "uniform float u_time;",
    "float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}",
    "float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);",
    "float a=hash(i),b=hash(i+vec2(1.0,0.0)),c=hash(i+vec2(0.0,1.0)),d=hash(i+vec2(1.0,1.0));",
    "return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);}",
    "void main(){",
    "vec2 uv=gl_FragCoord.xy/u_res.xy;",
    "float t=u_time*0.06;",
    "float n=noise(uv*3.0+vec2(t,t*0.7));",
    "float w=sin(uv.x*3.0+t)*0.5+0.5;",
    "float h=sin(uv.y*2.5-t*1.3+n*3.14)*0.5+0.5;",
    "vec3 c1=vec3(0.03,0.05,0.14);",
    "vec3 c2=vec3(0.16,0.10,0.45);",
    "vec3 c3=vec3(0.05,0.35,0.85);",
    "vec3 col=mix(mix(c1,c2,h),c3,w*n);",
    "col*=0.92-smoothstep(0.4,1.0,n)*0.2;",
    "gl_FragColor=vec4(col,1.0);",
    "}"
  ].join("");

  function compile(gl, type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }
  function initShader() {
    var cv = $("bg-canvas");
    if (!cv) return false;
    var gl = cv.getContext("webgl2") || cv.getContext("webgl") || cv.getContext("experimental-webgl");
    if (!gl) return false;
    var vs = compile(gl, gl.VERTEX_SHADER, VERT);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return false;
    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;
    gl.useProgram(prog);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
    ]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    shader.gl = gl;
    shader.prog = prog;
    shader.uTime = gl.getUniformLocation(prog, "u_time");
    shader.uRes = gl.getUniformLocation(prog, "u_res");
    shader.canvas = cv;
    resizeShader();
    return true;
  }
  function resizeShader() {
    var gl = shader.gl, cv = shader.canvas;
    if (!gl || !cv) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.floor(window.innerWidth * dpr);
    cv.height = Math.floor(window.innerHeight * dpr);
    gl.viewport(0, 0, cv.width, cv.height);
  }
  function renderShader() {
    var gl = shader.gl;
    if (!gl) return;
    gl.uniform1f(shader.uTime, shader.t);
    gl.uniform2f(shader.uRes, shader.canvas.width, shader.canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    shader.t += 1.0 / 60.0;
    shader.raf = requestAnimationFrame(renderShader);
  }
  function startShader() {
    if (REDUCED) return;              // 减弱动态效果：不启动 RAF
    if (!shader.gl && !initShader()) return false;
    if (!shader.raf) renderShader();
    return true;
  }
  function stopShader() {
    if (shader.raf) { cancelAnimationFrame(shader.raf); shader.raf = 0; }
  }

  // ---------- 主题应用 ----------
  function clearBodyBg() {
    document.body.style.background = "";
    document.body.style.backgroundImage = "";
  }
  function applyTheme(theme, bg) {
    document.documentElement.dataset.theme = theme;
    stopShader();
    clearBodyBg();
    if (theme === "light") {
      // 亮色：每次进入生成随机渐变
      document.body.style.background = randomGradient();
      document.body.style.backgroundAttachment = "fixed";
    } else if (theme === "dark") {
      // 暗色：动态着色器；启动失败或 reduced-motion 走静态深色渐变兜底
      var ok = startShader();
      if (!ok) {
        document.body.style.background =
          "radial-gradient(1200px 800px at 20% 10%, #1a1f4a, #0b1020 60%), linear-gradient(135deg,#0b1020,#1a0b2e)";
        document.body.style.backgroundAttachment = "fixed";
      }
    } else if (theme === "custom") {
      var url = bg || "";
      if (url) {
        document.body.style.background =
          "url('" + url.replace(/'/g, "%27") + "') center/cover no-repeat fixed";
      } else {
        document.body.style.background = randomGradient();
        document.body.style.backgroundAttachment = "fixed";
      }
    }
  }

  // ---------- 齿轮 + 设置面板 ----------
  function wireSettings() {
    var gear = $("gear-btn");
    var panel = $("settings-panel");
    if (!gear || !panel) return;

    function toggle() {
      var open = panel.hidden;
      panel.hidden = !open;
      gear.classList.toggle("open", open);
    }
    gear.addEventListener("click", function (e) { e.stopPropagation(); toggle(); });
    // 面板外点击关闭
    document.addEventListener("click", function (e) {
      if (!panel.hidden && !panel.contains(e.target) && e.target !== gear) {
        panel.hidden = true;
        gear.classList.remove("open");
      }
    });

    // 账号
    var setUser = $("set-user");
    if (setUser) {
      setUser.value = localStorage.getItem(LS_USER) || "u-demo";
      setUser.addEventListener("change", function () {
        var v = setUser.value.trim() || "u-demo";
        localStorage.setItem(LS_USER, v);
        location.search = "?user=" + encodeURIComponent(v);
      });
    }

    // 默认检索模式
    var setMode = $("set-retrieval-mode");
    var modeSel = $("retrieval-mode");
    if (setMode && modeSel) {
      var savedMode = localStorage.getItem(LS_MODE) || "hybrid";
      setMode.value = savedMode;
      modeSel.value = savedMode;
      setMode.addEventListener("change", function () {
        localStorage.setItem(LS_MODE, setMode.value);
        if (modeSel) modeSel.value = setMode.value;
      });
      // 同步：主页检索模式变化也回写持久化
      if (modeSel) {
        modeSel.addEventListener("change", function () {
          localStorage.setItem(LS_MODE, modeSel.value);
          if (setMode) setMode.value = modeSel.value;
        });
      }
    }

    // 主题单选
    var savedTheme = localStorage.getItem(LS_THEME) || "light";
    var radios = document.querySelectorAll('input[name="theme"]');
    radios.forEach(function (r) { if (r.value === savedTheme) r.checked = true; });
    applyTheme(savedTheme, localStorage.getItem(LS_BG));
    radios.forEach(function (r) {
      r.addEventListener("change", function () {
        var th = r.value;
        localStorage.setItem(LS_THEME, th);
        applyTheme(th, localStorage.getItem(LS_BG));
      });
    });

    // 自定义背景：URL 输入 + 本地文件
    var bgUrl = $("set-bg-url");
    var bgFile = $("set-bg-file");
    var bgBtn = $("set-bg-btn");
    if (bgUrl) {
      // 仅回显 http(s) URL；dataURL（本地文件上传）不塞进输入框以免拥挤
      var storedBg = localStorage.getItem(LS_BG) || "";
      bgUrl.value = /^https?:/i.test(storedBg) ? storedBg : "";
      bgUrl.addEventListener("change", function () {
        var v = bgUrl.value.trim();
        if (v) localStorage.setItem(LS_BG, v);
        selectTheme("custom");
        applyTheme("custom", v);
      });
    }
    if (bgBtn && bgFile) {
      bgBtn.addEventListener("click", function () { bgFile.click(); });
      bgFile.addEventListener("change", function () {
        var f = bgFile.files && bgFile.files[0];
        if (!f) return;
        var reader = new FileReader();
        reader.onload = function () {
          var data = String(reader.result || "");
          localStorage.setItem(LS_BG, data);
          if (bgUrl) bgUrl.value = "";
          selectTheme("custom");
          applyTheme("custom", data);
        };
        reader.readAsDataURL(f);
      });
    }
  }
  function selectTheme(theme) {
    document.querySelectorAll('input[name="theme"]').forEach(function (r) {
      r.checked = r.value === theme;
    });
    localStorage.setItem(LS_THEME, theme);
  }

  // ---------- 初始化 ----------
  function init() {
    wireSettings();
    bindSpotlightLeave();
    // 窗口尺寸变化重设着色器视口
    window.addEventListener("resize", function () {
      if (document.documentElement.dataset.theme === "dark") resizeShader();
    });
    // 页面隐藏时暂停 RAF（节流），可见时恢复
    document.addEventListener("visibilitychange", function () {
      if (document.documentElement.dataset.theme !== "dark") return;
      if (document.hidden) stopShader();
      else startShader();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
