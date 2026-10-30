"use strict";

/**
 * WebGL2 viewer for the buffers written by `demo/build.py`.
 *
 * No dependencies and no build step: there is no npm in this project, so
 * everything here is vanilla ES2020 against the raw WebGL2 API.
 *
 * The one structural decision worth knowing: nothing is a vertex buffer.
 * Positions, per-node attributes and the edge list all live in textures, and
 * both draw passes synthesise their geometry from `gl_VertexID`. That means
 * highlighting a closure re-uploads a 4-byte-per-node attribute texture
 * instead of rebuilding a 160MB edge buffer, and the edge list is uploaded
 * progressively without any buffer juggling.
 *
 * Node indices travel through float textures, which is exact for indices
 * below 2^24 -- about 16.7M nodes, far past anything lean4export produces.
 */

const DATA = "data";
const TEX_W = 4096; // texels per row in every data texture
const EDGE_UPLOAD_ROWS = 64; // edge texture rows pushed per frame
const PICK_GRID = 512;
const PICK_PX = 14; // hover radius in device-independent pixels

const KIND_COLORS = [
  [0.98, 0.55, 0.35],
  [0.45, 0.72, 0.98],
  [0.62, 0.86, 0.5],
  [0.93, 0.78, 0.4],
  [0.78, 0.6, 0.95],
  [0.5, 0.85, 0.82],
  [0.95, 0.6, 0.72],
  [0.7, 0.7, 0.78],
  [0.55, 0.62, 0.95],
  [0.85, 0.85, 0.55],
  [0.6, 0.9, 0.7],
  [0.9, 0.7, 0.55],
  [0.75, 0.55, 0.7],
  [0.55, 0.78, 0.9],
  [0.88, 0.5, 0.5],
  [0.65, 0.65, 0.65],
];

// Index order is the `u_colorMode` contract in NODE_VS, and the option order
// in index.html's `colorby` select.
const COLOR_MODES = ["kind", "community", "depth"];

const HL_NONE = 0;
const HL_FWD = 1;
const HL_REV = 2;
const HL_SELF = 3;

const $ = (id) => document.getElementById(id);

const VERT_COMMON = `#version 300 es
precision highp float;

uniform sampler2D u_pos;    // rgba32f: x, y, depth, 0
uniform sampler2D u_attr;   // rgba8:   kind, flags, highlight, 0
uniform int u_texW;
uniform vec2 u_center;
uniform float u_zoom;
uniform float u_aspect;
uniform float u_depthMin;
uniform bool u_showInternal;

ivec2 texel(int i) { return ivec2(i % u_texW, i / u_texW); }
vec4 nodePos(int i) { return texelFetch(u_pos, texel(i), 0); }
vec4 nodeAttr(int i) { return texelFetch(u_attr, texel(i), 0) * 255.0; }
vec2 project(vec2 p) {
  vec2 v = (p - u_center) * u_zoom;
  v.x /= u_aspect;
  return v;
}
bool bitSet(float flags, float bit) { return mod(floor(flags / bit), 2.0) >= 0.5; }
bool culled(vec4 pos, vec4 attr) {
  if (!u_showInternal && bitSet(attr.g, 1.0)) return true;
  return pos.z < u_depthMin;
}
`;

const NODE_VS =
  VERT_COMMON +
  `
uniform float u_pointSize;
uniform bool u_hasSelection;
uniform vec3 u_palette[16];
uniform int u_colorMode;    // 0 kind, 1 community, 2 kernel depth
uniform float u_depthMax;

out vec4 v_color;

// Kernel depth is one scalar per node, so it belongs in a visual channel
// rather than in a coordinate -- giving it an axis is what pinned the old
// layout to 134 horizontal lines.  The u_pos blue channel already holds the
// depth, so the ramp costs no buffer and no upload, and being continuous it
// does not band across those 134 levels the way 16 palette slots would.
vec3 depthRamp(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 low = vec3(0.25, 0.42, 0.85);   // depth 0: the foundations
  vec3 mid = vec3(0.30, 0.80, 0.70);
  vec3 high = vec3(0.98, 0.85, 0.35);  // deepest: everything built on them
  return t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, t * 2.0 - 1.0);
}

void main() {
  int id = gl_VertexID;
  vec4 pos = nodePos(id);
  vec4 attr = nodeAttr(id);

  if (culled(pos, attr)) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_color = vec4(0.0);
    return;
  }

  float hl = floor(attr.b + 0.5);
  vec3 base = u_colorMode == 2
    ? depthRamp(pos.z / max(u_depthMax, 1.0))
    : u_palette[int(floor(attr.r + 0.5)) & 15];
  float size = u_pointSize;
  float alpha = 0.85;

  if (hl == ${HL_SELF}.0) {
    base = vec3(1.0);
    size *= 3.0;
  } else if (hl == ${HL_FWD}.0) {
    base = vec3(1.0, 0.65, 0.34);
    size *= 1.7;
  } else if (hl == ${HL_REV}.0) {
    base = vec3(0.34, 0.83, 0.87);
    size *= 1.7;
  } else if (u_hasSelection) {
    base *= 0.35;
    alpha = 0.3;
  }

  gl_Position = vec4(project(pos.xy), 0.0, 1.0);
  gl_PointSize = max(size, 1.0);
  v_color = vec4(base, alpha);
}
`;

const NODE_FS = `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 fragColor;

void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = dot(d, d);
  if (r > 0.25) discard;
  float edge = smoothstep(0.25, 0.16, r);
  fragColor = vec4(v_color.rgb, v_color.a * edge);
}
`;

const EDGE_VS =
  VERT_COMMON +
  `
uniform sampler2D u_edge;   // rg32f: src, dst as float-exact indices
uniform int u_edgeTexW;
uniform float u_edgeAlpha;
uniform bool u_hasSelection;

out vec4 v_color;

void main() {
  int eid = gl_VertexID >> 1;
  vec2 e = texelFetch(u_edge, ivec2(eid % u_edgeTexW, eid / u_edgeTexW), 0).xy;
  int src = int(e.x);
  int dst = int(e.y);
  int self = (gl_VertexID & 1) == 0 ? src : dst;

  vec4 posA = nodePos(src), posB = nodePos(dst);
  vec4 attrA = nodeAttr(src), attrB = nodeAttr(dst);

  if (culled(posA, attrA) || culled(posB, attrB)) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    v_color = vec4(0.0);
    return;
  }

  float ha = floor(attrA.b + 0.5), hb = floor(attrB.b + 0.5);
  bool fwd = (ha == ${HL_FWD}.0 || ha == ${HL_SELF}.0) &&
             (hb == ${HL_FWD}.0 || hb == ${HL_SELF}.0);
  bool rev = (ha == ${HL_REV}.0 || ha == ${HL_SELF}.0) &&
             (hb == ${HL_REV}.0 || hb == ${HL_SELF}.0);

  vec3 color = vec3(0.42, 0.52, 0.72);
  float alpha = u_edgeAlpha;

  if (u_hasSelection) {
    if (fwd) {
      color = vec3(1.0, 0.65, 0.34);
      alpha = 0.55;
    } else if (rev) {
      color = vec3(0.34, 0.83, 0.87);
      alpha = 0.55;
    } else {
      alpha = u_edgeAlpha * 0.15;
    }
  }

  gl_Position = vec4(project(nodePos(self).xy), 0.0, 1.0);
  v_color = vec4(color, alpha);
}
`;

const EDGE_FS = `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }
`;

// --- data loading -----------------------------------------------------------

async function fetchTyped(name, Kind) {
  const res = await fetch(`${DATA}/${name}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
  const buf = await res.arrayBuffer();
  return new Kind(buf);
}

async function loadAll(say) {
  say("fetching meta.json…");
  const metaRes = await fetch(`${DATA}/meta.json`, { cache: "no-cache" });
  if (!metaRes.ok) {
    throw new Error(
      `no data found at ${DATA}/meta.json — run demo/build.py first (see demo/README.md)`,
    );
  }
  const meta = await metaRes.json();

  const want = [
    ["nodes", "nodes.f32", Float32Array],
    ["kind", "node_kind.u8", Uint8Array],
    ["flags", "node_flags.u8", Uint8Array],
    ["depth", "node_depth.u16", Uint16Array],
    ["edges", "edges.u32", Uint32Array],
    ["edgeOrder", "edge_order.u32", Uint32Array],
    ["comm", "node_comm.u16", Uint16Array],
    ["fwdPtr", "fwd_indptr.u32", Uint32Array],
    ["fwdIdx", "fwd_indices.u32", Uint32Array],
    ["revPtr", "rev_indptr.u32", Uint32Array],
    ["revIdx", "rev_indices.u32", Uint32Array],
    ["nameBlob", "names.bin", Uint8Array],
    ["nameIdx", "names_idx.u32", Uint32Array],
    ["nameOrder", "name_order.u32", Uint32Array],
    ["memberBlob", "members.bin", Uint8Array],
    ["memberIdx", "members_idx.u32", Uint32Array],
  ];

  const out = { meta };
  let done = 0;
  await Promise.all(
    want.map(async ([key, file, Kind]) => {
      out[key] = await fetchTyped(file, Kind);
      say(`fetched ${++done}/${want.length} buffers…`);
    }),
  );
  return out;
}

// --- GL helpers -------------------------------------------------------------

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`shader: ${gl.getShaderInfoLog(sh)}`);
  }
  return sh;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
  }
  const loc = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, "");
    loc[name] = gl.getUniformLocation(p, name);
  }
  return { p, loc };
}

function makeTexture(gl, unit, internal, format, type, width, height, data) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, data ?? null);
  return tex;
}

const rowsFor = (count, width) => Math.max(1, Math.ceil(count / width));

// --- the viewer -------------------------------------------------------------

class Viewer {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.d = data;
    this.n = data.meta.nodes;
    this.totalEdges = data.meta.edges;
    this.maxDepth = data.meta.max_depth;

    this.center = [0, 0];
    this.zoom = 1;
    this.depthMin = 0;
    this.showInternal = true;
    this.showEdges = true;
    this.selected = -1;
    this.hover = -1;
    // The build ranks edges by how much they explain; `drawn_edges` is where
    // it put the line between structure and redundancy.  Older data
    // directories have no such key, so fall back to the previous default.
    this.edgeBudget = Math.min(this.totalEdges, data.meta.drawn_edges ?? 1 << 21);
    this.colorBy = "depth";

    this.names = new Array(this.n);
    this.decoder = new TextDecoder();

    this.gl = canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      powerPreference: "high-performance",
    });
    if (!this.gl) throw new Error("WebGL2 is unavailable in this browser");

    this.initGL();
    this.buildPickGrid();
    this.fit();
  }

  initGL() {
    const gl = this.gl;
    this.nodeProg = program(gl, NODE_VS, NODE_FS);
    this.edgeProg = program(gl, EDGE_VS, EDGE_FS);
    this.vao = gl.createVertexArray();

    const rows = rowsFor(this.n, TEX_W);

    // rgba32f: x, y, depth, 0
    const pos = new Float32Array(TEX_W * rows * 4);
    for (let i = 0; i < this.n; i++) {
      pos[i * 4] = this.d.nodes[i * 2];
      pos[i * 4 + 1] = this.d.nodes[i * 2 + 1];
      pos[i * 4 + 2] = this.d.depth[i];
    }
    this.posTex = makeTexture(gl, 0, gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_W, rows, pos);

    // rgba8: kind, flags, highlight, 0
    this.attr = new Uint8Array(TEX_W * rows * 4);
    for (let i = 0; i < this.n; i++) {
      this.attr[i * 4] = this.d.kind[i];
      this.attr[i * 4 + 1] = this.d.flags[i];
    }
    this.attrRows = rows;
    this.attrTex = makeTexture(
      gl,
      1,
      gl.RGBA8,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      TEX_W,
      rows,
      this.attr,
    );

    this.allocEdgeTexture();
    gl.clearColor(0.043, 0.051, 0.071, 1);
  }

  /** Allocate (not fill) the edge texture; rows stream in during render. */
  allocEdgeTexture() {
    const gl = this.gl;
    this.edgeRows = rowsFor(this.edgeBudget, TEX_W);
    this.uploadedRows = 0;

    if (this.edgeTex) gl.deleteTexture(this.edgeTex);
    this.edgeTex = makeTexture(
      gl,
      2,
      gl.RG32F,
      gl.RG,
      gl.FLOAT,
      TEX_W,
      this.edgeRows,
      null,
    );
    this.rowScratch = new Float32Array(TEX_W * EDGE_UPLOAD_ROWS * 2);
  }

  /**
   * Push the next slice of edges into the texture.
   *
   * The budget takes a *prefix* of `edge_order`, which the build sorted so
   * that the edges carrying structure come first and the ones implied by a
   * two-step path come last.  A prefix of a ranking beats the stride sample
   * this used to take: at any budget you get the most informative edges
   * rather than an even scattering, and the picture is still reproducible
   * because the ranking is deterministic.
   */
  pumpEdgeUpload() {
    if (this.uploadedRows >= this.edgeRows) return false;
    const gl = this.gl;
    const rows = Math.min(EDGE_UPLOAD_ROWS, this.edgeRows - this.uploadedRows);
    const first = this.uploadedRows * TEX_W;
    const scratch = this.rowScratch;

    for (let i = 0; i < rows * TEX_W; i++) {
      const slot = first + i;
      // Past the budget the texture is padded with a degenerate edge rather
      // than left undefined; `drawable` in render() keeps it off screen.
      const base = slot < this.edgeBudget ? this.d.edgeOrder[slot] * 2 : 0;
      scratch[i * 2] = this.d.edges[base];
      scratch[i * 2 + 1] = this.d.edges[base + 1];
    }

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.edgeTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      this.uploadedRows,
      TEX_W,
      rows,
      gl.RG,
      gl.FLOAT,
      scratch.subarray(0, rows * TEX_W * 2),
    );
    this.uploadedRows += rows;
    return true;
  }

  // --- names ----------------------------------------------------------------

  name(i) {
    let hit = this.names[i];
    if (hit === undefined) {
      hit = this.decoder.decode(
        this.d.nameBlob.subarray(this.d.nameIdx[i], this.d.nameIdx[i + 1]),
      );
      this.names[i] = hit;
    }
    return hit;
  }

  members(i) {
    const text = this.decoder.decode(
      this.d.memberBlob.subarray(this.d.memberIdx[i], this.d.memberIdx[i + 1]),
    );
    return text ? text.split("\n") : [];
  }

  /** Prefix search over the shipped sorted permutation. Case-sensitive, as Lean names are. */
  search(prefix, limit = 25) {
    if (!prefix) return [];
    const order = this.d.nameOrder;
    let lo = 0;
    let hi = order.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.name(order[mid]) < prefix) lo = mid + 1;
      else hi = mid;
    }
    const out = [];
    for (let i = lo; i < order.length && out.length < limit; i++) {
      const node = order[i];
      if (!this.name(node).startsWith(prefix)) break;
      out.push(node);
    }
    return out;
  }

  // --- picking --------------------------------------------------------------

  buildPickGrid() {
    const n = this.n;
    const cells = PICK_GRID * PICK_GRID;
    const counts = new Uint32Array(cells + 1);
    const cellOf = new Uint32Array(n);

    for (let i = 0; i < n; i++) {
      const gx = Math.min(PICK_GRID - 1, Math.max(0, ((this.d.nodes[i * 2] + 1) / 2) * PICK_GRID));
      const gy = Math.min(
        PICK_GRID - 1,
        Math.max(0, ((this.d.nodes[i * 2 + 1] + 1) / 2) * PICK_GRID),
      );
      const cell = (gy | 0) * PICK_GRID + (gx | 0);
      cellOf[i] = cell;
      counts[cell + 1]++;
    }
    for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];

    const items = new Uint32Array(n);
    const cursor = counts.slice(0, cells);
    for (let i = 0; i < n; i++) items[cursor[cellOf[i]]++] = i;

    this.pick = { start: counts, items };
  }

  nodeAt(worldX, worldY, radius) {
    const { start, items } = this.pick;
    const toCell = (v) => Math.min(PICK_GRID - 1, Math.max(0, (((v + 1) / 2) * PICK_GRID) | 0));
    const x0 = toCell(worldX - radius);
    const x1 = toCell(worldX + radius);
    const y0 = toCell(worldY - radius);
    const y1 = toCell(worldY + radius);

    let best = -1;
    let bestDist = radius * radius;
    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const cell = gy * PICK_GRID + gx;
        for (let k = start[cell]; k < start[cell + 1]; k++) {
          const i = items[k];
          if (!this.showInternal && this.d.flags[i] & 1) continue;
          if (this.d.depth[i] < this.depthMin) continue;
          const dx = this.d.nodes[i * 2] - worldX;
          const dy = this.d.nodes[i * 2 + 1] - worldY;
          const dist = dx * dx + dy * dy;
          if (dist < bestDist) {
            bestDist = dist;
            best = i;
          }
        }
      }
    }
    return best;
  }

  // --- closures -------------------------------------------------------------

  /** Iterative BFS over a shipped CSR. The library's `closure` / `rclosure`, in the browser. */
  reach(root, indptr, indices) {
    const seen = new Uint8Array(this.n);
    const queue = new Uint32Array(this.n);
    let head = 0;
    let tail = 0;
    queue[tail++] = root;
    seen[root] = 1;

    while (head < tail) {
      const node = queue[head++];
      for (let k = indptr[node]; k < indptr[node + 1]; k++) {
        const next = indices[k];
        if (!seen[next]) {
          seen[next] = 1;
          queue[tail++] = next;
        }
      }
    }
    return { seen, count: tail - 1 };
  }

  select(node) {
    this.selected = node;
    const hl = this.attr;
    for (let i = 0; i < this.n; i++) hl[i * 4 + 2] = HL_NONE;

    let fwd = 0;
    let rev = 0;
    let direct = 0;

    if (node >= 0) {
      const f = this.reach(node, this.d.fwdPtr, this.d.fwdIdx);
      const r = this.reach(node, this.d.revPtr, this.d.revIdx);
      fwd = f.count;
      rev = r.count;
      direct = this.d.fwdPtr[node + 1] - this.d.fwdPtr[node];
      for (let i = 0; i < this.n; i++) {
        if (f.seen[i]) hl[i * 4 + 2] = HL_FWD;
        else if (r.seen[i]) hl[i * 4 + 2] = HL_REV;
      }
      hl[node * 4 + 2] = HL_SELF;
    }

    this.uploadAttr();
    return { fwd, rev, direct };
  }

  /** Re-push the whole per-node attribute texture (kind, flags, highlight). */
  uploadAttr() {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.attrTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      TEX_W,
      this.attrRows,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      this.attr,
    );
  }

  /**
   * Recolour by declaration kind, by community band, or by kernel depth.
   *
   * The first two are palette indices in the attribute texture's red channel,
   * so they are a CPU rewrite and one upload -- the same path `select` already
   * takes for highlighting.  Bands wrap at 16 palette slots; they are
   * spatially separated, so two distant bands sharing a colour reads
   * acceptably.  Depth is a shader-side ramp over the depth the position
   * texture already carries, so it rewrites nothing.
   */
  setColorBy(mode) {
    this.colorBy = mode;
    if (mode === "depth") return;
    const byKind = mode !== "community";
    for (let i = 0; i < this.n; i++) {
      this.attr[i * 4] = byKind ? this.d.kind[i] : this.d.comm[i] % 16;
    }
    this.uploadAttr();
  }

  // --- camera ---------------------------------------------------------------

  get aspect() {
    return this.canvas.width / this.canvas.height;
  }

  fit() {
    this.center = [0, 0];
    this.zoom = 0.95;
  }

  screenToWorld(px, py) {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((px - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((py - rect.top) / rect.height) * 2;
    return [this.center[0] + (ndcX * this.aspect) / this.zoom, this.center[1] + ndcY / this.zoom];
  }

  worldRadius(pixels) {
    const rect = this.canvas.getBoundingClientRect();
    return (pixels / rect.height) * 2 / this.zoom;
  }

  // --- render ---------------------------------------------------------------

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  setCommon(loc) {
    const gl = this.gl;
    gl.uniform1i(loc.u_pos, 0);
    gl.uniform1i(loc.u_attr, 1);
    gl.uniform1i(loc.u_texW, TEX_W);
    gl.uniform2f(loc.u_center, this.center[0], this.center[1]);
    gl.uniform1f(loc.u_zoom, this.zoom);
    gl.uniform1f(loc.u_aspect, this.aspect);
    gl.uniform1f(loc.u_depthMin, this.depthMin);
    gl.uniform1i(loc.u_showInternal, this.showInternal ? 1 : 0);
    gl.uniform1i(loc.u_hasSelection, this.selected >= 0 ? 1 : 0);
  }

  render() {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindVertexArray(this.vao);
    gl.enable(gl.BLEND);

    const drawable = Math.min(this.edgeBudget, this.uploadedRows * TEX_W);

    if (this.showEdges && drawable > 0) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive: density reads as shading
      gl.useProgram(this.edgeProg.p);
      const loc = this.edgeProg.loc;
      this.setCommon(loc);
      gl.uniform1i(loc.u_edge, 2);
      gl.uniform1i(loc.u_edgeTexW, TEX_W);
      // Fainter the more there is, so the overview is a density field rather
      // than a solid mass.
      gl.uniform1f(loc.u_edgeAlpha, Math.max(0.012, Math.min(0.5, 24000 / drawable)));

      const CHUNK = 1 << 20;
      for (let first = 0; first < drawable; first += CHUNK) {
        const count = Math.min(CHUNK, drawable - first);
        gl.drawArrays(gl.LINES, first * 2, count * 2);
      }
    }

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.nodeProg.p);
    const loc = this.nodeProg.loc;
    this.setCommon(loc);
    gl.uniform1f(loc.u_pointSize, Math.max(1.5, Math.min(14, 2.2 * Math.sqrt(this.zoom))));
    gl.uniform3fv(loc.u_palette, new Float32Array(KIND_COLORS.flat()));
    gl.uniform1i(loc.u_colorMode, COLOR_MODES.indexOf(this.colorBy));
    gl.uniform1f(loc.u_depthMax, this.maxDepth);
    gl.drawArrays(gl.POINTS, 0, this.n);

    return drawable;
  }
}

// --- wiring -----------------------------------------------------------------

const fmt = (x) => x.toLocaleString("en-US");

function humanEdges(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

async function main() {
  const canvas = $("gl");
  const boot = $("boot");
  const say = (m) => ($("bootmsg").textContent = m);

  let data;
  let view;
  try {
    data = await loadAll(say);
    say("building textures…");
    view = new Viewer(canvas, data);
  } catch (err) {
    say(String(err.message || err));
    $("bootmsg").style.color = "#ff8a8a";
    return;
  }
  boot.classList.add("done");

  const meta = data.meta;
  $("s-nodes").textContent = fmt(meta.nodes);
  $("s-edges").textContent =
    meta.drawn_edges === undefined || meta.drawn_edges === meta.edges
      ? fmt(meta.edges)
      : `${fmt(meta.drawn_edges)} of ${fmt(meta.edges)}`;
  $("s-depth").textContent = `0–${meta.max_depth}`;
  $("prov").textContent = meta.lean_version ? `· Lean ${meta.lean_version}` : "";

  const swatch = (label, i) => {
    const c = KIND_COLORS[i % 16].map((v) => Math.round(v * 255)).join(",");
    return `<span><i style="background:rgb(${c})"></i>${label}</span>`;
  };
  const drawLegend = () => {
    if (view.colorBy === "depth") {
      const ramp = "linear-gradient(90deg, rgb(64,107,217), rgb(77,204,179), rgb(250,217,89))";
      $("legend").innerHTML =
        `<span>colour = kernel depth` +
        `<b style="width:120px;height:9px;border-radius:5px;background:${ramp}"></b>` +
        `0 &rarr; ${meta.max_depth}</span>`;
      return;
    }
    if (view.colorBy === "community") {
      const n = meta.band_count ?? 0;
      $("legend").innerHTML =
        `<span>colour = community band (${n}) &mdash; position is the layout, not the id</span>`;
      return;
    }
    $("legend").innerHTML = meta.kinds.map(swatch).join("");
  };
  $("colorby").value = view.colorBy;
  drawLegend();

  const budget = $("budget");
  budget.max = String(Math.ceil(Math.log2(Math.max(meta.edges, 2))));
  budget.value = String(Math.round(Math.log2(view.edgeBudget)));
  const showBudget = () => {
    $("budgetval").textContent = `${humanEdges(view.edgeBudget)}/${humanEdges(meta.edges)}`;
  };
  showBudget();

  const depthmin = $("depthmin");
  depthmin.max = String(meta.max_depth);
  $("depthval").textContent = "0";

  // Rebuilding the edge texture re-streams it, so react on release, not drag.
  budget.addEventListener("change", () => {
    view.edgeBudget = Math.min(meta.edges, 2 ** Number(budget.value));
    view.allocEdgeTexture();
    showBudget();
  });
  depthmin.addEventListener("input", () => {
    view.depthMin = Number(depthmin.value);
    $("depthval").textContent = depthmin.value;
  });
  $("showinternal").addEventListener("change", (e) => {
    view.showInternal = e.target.checked;
  });
  $("colorby").addEventListener("change", (e) => {
    view.setColorBy(e.target.value);
    drawLegend();
  });

  const selection = $("selection");
  const applySelection = (node) => {
    const counts = view.select(node);
    if (node < 0) {
      selection.classList.remove("on");
      return;
    }
    selection.classList.add("on");
    $("selname").textContent = view.name(node);
    $("s-fwd").textContent = fmt(counts.fwd);
    $("s-rev").textContent = fmt(counts.rev);
    $("s-direct").textContent = fmt(counts.direct);
  };

  const results = $("results");
  $("search").addEventListener("input", (e) => {
    const hits = view.search(e.target.value.trim());
    results.innerHTML = "";
    for (const node of hits) {
      const btn = document.createElement("button");
      btn.textContent = view.name(node);
      btn.addEventListener("click", () => {
        view.center = [data.nodes[node * 2], data.nodes[node * 2 + 1]];
        view.zoom = Math.max(view.zoom, 40);
        applySelection(node);
      });
      results.appendChild(btn);
    }
  });

  // --- pointer ---------------------------------------------------------------
  let dragging = false;
  let last = [0, 0];
  let moved = 0;

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    moved = 0;
    last = [e.clientX, e.clientY];
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.classList.remove("dragging");
    canvas.releasePointerCapture(e.pointerId);
    if (moved < 5) {
      const [wx, wy] = view.screenToWorld(e.clientX, e.clientY);
      applySelection(view.nodeAt(wx, wy, view.worldRadius(PICK_PX)));
    }
  });

  const tip = $("tip");
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) {
      moved += Math.abs(e.clientX - last[0]) + Math.abs(e.clientY - last[1]);
      const rect = canvas.getBoundingClientRect();
      view.center[0] -= (((e.clientX - last[0]) / rect.height) * 2 * view.aspect) / view.zoom;
      view.center[1] += (((e.clientY - last[1]) / rect.height) * 2) / view.zoom;
      last = [e.clientX, e.clientY];
      tip.classList.remove("on");
      return;
    }

    const [wx, wy] = view.screenToWorld(e.clientX, e.clientY);
    const node = view.nodeAt(wx, wy, view.worldRadius(PICK_PX));
    if (node < 0) {
      tip.classList.remove("on");
      return;
    }
    const out = data.fwdPtr[node + 1] - data.fwdPtr[node];
    const into = data.revPtr[node + 1] - data.revPtr[node];
    const members = view.members(node);
    $("tipname").textContent = view.name(node);
    $("tipmeta").textContent =
      `${meta.kinds[data.kind[node]]} · depth ${data.depth[node]} · ` +
      `${out} out / ${into} in${members.length > 1 ? ` · block of ${members.length}` : ""}`;
    $("members").textContent = members.length > 1 ? members.join("\n") : "";
    tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 480)}px`;
    tip.style.top = `${e.clientY + 14}px`;
    tip.classList.add("on");
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [wx, wy] = view.screenToWorld(e.clientX, e.clientY);
    const factor = Math.exp(-e.deltaY * 0.0015);
    view.zoom = Math.max(0.2, Math.min(5000, view.zoom * factor));
    const [ax, ay] = view.screenToWorld(e.clientX, e.clientY);
    view.center[0] += wx - ax;
    view.center[1] += wy - ay;
  });

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.key === "f") view.fit();
    else if (e.key === "r") {
      view.fit();
      applySelection(-1);
    } else if (e.key === "e") view.showEdges = !view.showEdges;
    else if (e.key === "Escape") applySelection(-1);
  });

  // --- loop ------------------------------------------------------------------
  let frames = 0;
  let fpsAt = performance.now();

  const frame = () => {
    view.pumpEdgeUpload();
    const drawn = view.render();

    frames++;
    const now = performance.now();
    if (now - fpsAt > 500) {
      $("s-fps").textContent = String(Math.round((frames * 1000) / (now - fpsAt)));
      $("s-drawn").textContent =
        drawn < view.edgeBudget ? `${fmt(drawn)} (streaming)` : fmt(drawn);
      frames = 0;
      fpsAt = now;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main();
