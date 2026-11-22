// ElRobot mission control: 3D twin (URDF meshes posed by the real joint readings), detected objects, the
// table-camera -> gripper-camera fusion, and the reasoning stream. Scene units: cm, z up, base frame.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const $ = (s) => document.querySelector(s);
const api = async (p) => (await fetch(p, { cache: 'no-store' })).json();
const fmt = (v, d = 1) => (v >= 0 ? '+' : '') + v.toFixed(d);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const ACCENT = 0xd97757, TEAL = 0x2dd4bf, INFO = 0x5aa9ff, OK = 0x3ddc97;

// ------------------------------------------------------------------------------------------- renderer + scene
const view = $('#view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.className = 'labels';
view.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d12);
scene.fog = new THREE.Fog(0x0a0d12, 180, 380);
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 1200);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
const VIEWS = {
  orbit: [[40, 74, 44], [-2, 24, 8]], top: [[0, 24, 105], [0, 24, 0]],
  table: [[2, 82, 30], [0, 22, 8]], side: [[88, 24, 22], [0, 24, 9]],
};
let viewAnim = null;
function setView(name) {
  const [p, t] = VIEWS[name];
  viewAnim = { p0: camera.position.clone(), t0: controls.target.clone(), p1: new THREE.Vector3(...p),
               t1: new THREE.Vector3(...t), s: performance.now() };
  document.querySelectorAll('#viewBtns [data-v]').forEach((b) => b.classList.toggle('on', b.dataset.v === name));
}
camera.position.set(...VIEWS.orbit[0]);
controls.target.set(...VIEWS.orbit[1]);
controls.update();

const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x14171c, 0.85);
hemi.position.set(0, 0, 1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(45, 70, 130);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -80, right: 80, top: 80, bottom: -80, near: 10, far: 400 });
scene.add(sun);
const rim = new THREE.DirectionalLight(0x8fb4ff, 0.55);
rim.position.set(-70, -50, 50);
scene.add(rim);

// desk, grid, base-frame axes
const desk = new THREE.Mesh(new THREE.PlaneGeometry(260, 260), new THREE.MeshStandardMaterial({ color: 0x2a231d, roughness: 0.95 }));
desk.position.set(0, 30, -0.05);
desk.receiveShadow = true;
scene.add(desk);
const grid = new THREE.GridHelper(200, 40, 0x3a414d, 0x262c36);
grid.rotation.x = Math.PI / 2;
grid.position.set(0, 30, 0.01);
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);
const GIZMO = new THREE.Vector3(-44, 52, 0.2);  // base-frame axes, drawn in a corner away from the work area
function arrow(dir, color, text) {
  const a = new THREE.ArrowHelper(new THREE.Vector3(...dir), GIZMO, 9, color, 1.8, 1.1);
  scene.add(a);
  const l = new CSS2DObject(Object.assign(document.createElement('div'), { className: 'lbl axis', innerHTML: text }));
  l.position.set(GIZMO.x + dir[0] * 11, GIZMO.y + dir[1] * 11, 1);
  scene.add(l);
}
arrow([1, 0, 0], 0xff6b6b, '+x <small>arm\'s right</small>');
arrow([0, 1, 0], 0x6bff9e, '+y <small>away from base</small>');

// ------------------------------------------------------------------------------------------- the arm
const arm = new THREE.Group();
scene.add(arm);
const links = {};
const MAT_SERVO = new THREE.MeshStandardMaterial({ color: 0x1b1f26, metalness: 0.35, roughness: 0.45 });
const MAT_PRINT = new THREE.MeshStandardMaterial({ color: 0xe6e9ee, metalness: 0.05, roughness: 0.55 });
const MAT_JAW = new THREE.MeshStandardMaterial({ color: ACCENT, metalness: 0.1, roughness: 0.5 });
async function loadArm() {
  const manifest = await api('/api/meshes');
  await Promise.all(Object.entries(manifest).map(async ([name, m]) => {
    const buf = await (await fetch(m.url)).arrayBuffer();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf), 3));
    g.computeVertexNormals();
    const mat = name.startsWith('ST3215') ? MAT_SERVO : name.startsWith('Gripper_Jaw') ? MAT_JAW : MAT_PRINT;
    const mesh = new THREE.Mesh(g, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.visible = false;  // until the first pose arrives
    arm.add(mesh);
    links[name] = { mesh, p0: new THREE.Vector3(), q0: new THREE.Quaternion(), p1: null, q1: new THREE.Quaternion(),
                    cp: new THREE.Vector3(), cq: new THREE.Quaternion(), t0: 0 };
  }));
}
const ONE = new THREE.Vector3(1, 1, 1);
const tmpM = new THREE.Matrix4(), tmpS = new THREE.Vector3();
const tip = { p0: new THREE.Vector3(), p1: null, cur: new THREE.Vector3(), t0: 0, axis: new THREE.Vector3(0, 0, -1) };
const LERP_MS = 130;
function applyPose(r) {
  const now = performance.now();
  for (const [name, m] of Object.entries(r.links)) {
    const L = links[name];
    if (!L) continue;
    tmpM.set(...m);
    const p = new THREE.Vector3(), q = new THREE.Quaternion();
    tmpM.decompose(p, q, tmpS);
    if (!L.p1) { L.p0.copy(p); L.q0.copy(q); L.cp.copy(p); L.cq.copy(q); } else { L.p0.copy(L.cp); L.q0.copy(L.cq); }
    L.p1 = p; L.q1.copy(q); L.t0 = now;
    L.mesh.visible = true;
  }
  const tp = new THREE.Vector3(...r.tip);
  if (!tip.p1) tip.cur.copy(tp);
  tip.p0.copy(tip.cur); tip.p1 = tp; tip.t0 = now;
  tip.axis.set(...r.axis).normalize();
}

// fingertip marker, path trail, gripper axis ray
const tipDot = new THREE.Mesh(new THREE.SphereGeometry(0.7, 20, 12), new THREE.MeshBasicMaterial({ color: ACCENT }));
scene.add(tipDot);
const TRAIL_N = 1500;
const trailPos = new Float32Array(TRAIL_N * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setDrawRange(0, 0);
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.85 }));
trail.frustumCulled = false;
scene.add(trail);
let trailCount = 0;
const lastTrail = new THREE.Vector3(1e9, 0, 0);
function pushTrail(p) {
  if (p.distanceTo(lastTrail) < 0.3) return;
  lastTrail.copy(p);
  if (trailCount === TRAIL_N) { trailPos.copyWithin(0, 3); trailCount--; }
  trailPos.set([p.x, p.y, p.z], trailCount * 3);
  trailCount++;
  trailGeo.setDrawRange(0, trailCount);
  trailGeo.attributes.position.needsUpdate = true;
}
const rayGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const ray = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: INFO, dashSize: 1.2, gapSize: 0.8, transparent: true, opacity: 0.9 }));
ray.frustumCulled = false;
scene.add(ray);
const rayHit = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.2, 32), new THREE.MeshBasicMaterial({ color: INFO, transparent: true, opacity: 0.8 }));
scene.add(rayHit);

// ------------------------------------------------------------------------------------------- the mat
let matDone = false;
function drawMat(poly) {
  if (matDone || !poly || poly.length < 3) return;
  matDone = true;
  const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
  const mat = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0x1f2227, roughness: 1 }));
  mat.position.z = 0.04;
  mat.receiveShadow = true;
  scene.add(mat);
  const pts = poly.map(([x, y]) => new THREE.Vector3(x, y, 0.12));
  const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: OK, transparent: true, opacity: 0.7 }));
  scene.add(outline);
}

// ------------------------------------------------------------------------------------------- objects
function label(html, cls = 'lbl') {
  const d = document.createElement('div');
  d.className = cls;
  d.innerHTML = html;
  return new CSS2DObject(d);
}
function stdMat(color, extra = {}) { return new THREE.MeshStandardMaterial({ color, roughness: 0.45, ...extra }); }
function makeMug(body, handle) {
  const g = new THREE.Group();
  const bm = stdMat(body, { side: THREE.DoubleSide, transparent: true });
  const side = new THREE.Mesh(new THREE.CylinderGeometry(4.1, 3.7, 9.6, 48, 1, true), bm);
  side.rotation.x = Math.PI / 2; side.position.z = 4.8; side.castShadow = true;
  const bottom = new THREE.Mesh(new THREE.CircleGeometry(3.7, 40), bm);
  bottom.position.z = 0.4;
  const rimM = new THREE.Mesh(new THREE.TorusGeometry(4.05, 0.22, 10, 48), bm);
  rimM.position.z = 9.6;
  const hm = stdMat(handle, { transparent: true });
  const h = new THREE.Mesh(new THREE.TorusGeometry(2.4, 0.55, 12, 32, Math.PI), hm);
  h.rotation.x = Math.PI / 2;
  const hg = new THREE.Group();
  hg.add(h); hg.rotation.y = Math.PI / 2; hg.position.set(4.0, 0, 5.2);
  g.add(side, bottom, rimM, hg);
  g.userData.mats = [bm, hm];
  return g;
}
function stickerTex(color) {
  const c = document.createElement('canvas'); c.width = c.height = 96;
  const x = c.getContext('2d');
  x.fillStyle = '#0d0d0d'; x.fillRect(0, 0, 96, 96);
  x.fillStyle = color;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) x.fillRect(4 + i * 31, 4 + j * 31, 27, 27);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function makeCube() {
  const cols = ['#e53935', '#ff8f00', '#1e88e5', '#43a047', '#fafafa', '#fdd835'];
  const mats = cols.map((c) => new THREE.MeshStandardMaterial({ map: stickerTex(c), roughness: 0.5, transparent: true }));
  const m = new THREE.Mesh(new THREE.BoxGeometry(5.7, 5.7, 5.7), mats);
  m.position.z = 2.85; m.castShadow = true;
  const g = new THREE.Group(); g.add(m); g.userData.mats = mats;
  return g;
}
function makeBall() {
  const mat = stdMat(0xff8a3d, { roughness: 0.75, transparent: true });
  const m = new THREE.Mesh(new THREE.SphereGeometry(2.25, 32, 20), mat);
  m.position.z = 2.25; m.castShadow = true;
  const g = new THREE.Group(); g.add(m); g.userData.mats = [mat]; g.userData.ballMesh = m;
  return g;
}
const OBJ = {};
for (const [name, make] of [['white mug', () => makeMug(0xf4f4f2, 0xd9342b)], ['black mug', () => makeMug(0x15161a, 0xf0f0f0)],
                            ['rubiks cube', makeCube], ['orange ball', makeBall]]) {
  const g = make();
  g.visible = false;
  const l = label(name);
  l.position.set(0, 0, name.includes('mug') ? 12.5 : name === 'rubiks cube' ? 8 : 6);
  g.add(l);
  scene.add(g);
  OBJ[name] = { g, l, target: new THREE.Vector3(), seen: 0, lastXY: null, state: 'unseen' };
}
function setOpacity(o, a) { for (const m of o.g.userData.mats) m.opacity = a; }

// target ring + line to it, fusion markers, release marker
const ring = new THREE.Mesh(new THREE.RingGeometry(5.6, 6.4, 56), new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
ring.visible = false; scene.add(ring);
const toGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const toLine = new THREE.Line(toGeo, new THREE.LineDashedMaterial({ color: ACCENT, dashSize: 1.5, gapSize: 1 }));
toLine.frustumCulled = false; toLine.visible = false; scene.add(toLine);
const toLbl = label('', 'lbl dist'); scene.add(toLbl);
const fuse = new THREE.Group(); scene.add(fuse); fuse.visible = false;
const fTable = new THREE.Mesh(new THREE.RingGeometry(1.4, 1.9, 40), new THREE.MeshBasicMaterial({ color: 0x9aa4b5, side: THREE.DoubleSide }));
const fMeas = new THREE.Mesh(new THREE.CircleGeometry(1.6, 40), new THREE.MeshBasicMaterial({ color: TEAL, side: THREE.DoubleSide }));
const fGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const fLine = new THREE.Line(fGeo, new THREE.LineBasicMaterial({ color: TEAL }));
const fLblT = label('', 'lbl below'), fLblM = label('', 'lbl fuse');
fuse.add(fTable, fMeas, fLine, fLblT, fLblM);
const relMark = new THREE.Group(); scene.add(relMark); relMark.visible = false;
const relX = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.9, 24), new THREE.MeshBasicMaterial({ color: OK, side: THREE.DoubleSide }));
const relL = label('', 'lbl'); relMark.add(relX, relL);

// ------------------------------------------------------------------------------------------- state from the server
let S = { robot: null, scene: null, mode: 'live', robotOk: false };
let follow = false;
async function pollRobot() {
  try {
    const r = await api('/api/robot');
    S.mode = r.mode; S.robotOk = r.ok; S.robotErr = r.error;
    if (r.robot) { S.robot = r.robot; applyPose(r.robot); hud(r.robot); }
  } catch (e) { S.robotOk = false; }
  pills();
  setTimeout(pollRobot, 100);
}
async function pollScene() {
  try { S.scene = await api('/api/scene'); drawMat(S.scene.mat); updateObjects(); story(); } catch (e) { /* retry */ }
  setTimeout(pollScene, 250);
}

function pills() {
  const set = (id, cls, text) => { const p = $(id); p.className = 'pill ' + cls; p.innerHTML = '<i></i>' + text; };
  const sim = S.mode === 'sim';
  set('#pMode', sim ? 'sim' : 'ok', sim ? 'SIMULATION' : 'LIVE');
  $('#simBanner').style.display = sim ? 'block' : 'none';
  set('#pStation', S.robotOk ? 'ok' : 'bad', S.robotOk ? (sim ? 'SIM ARM' : 'STATION') : 'STATION OFFLINE');
  const r = S.robot, errs = r ? r.errors.filter(Boolean).length : 0;
  set('#pMotors', !r ? '' : errs ? 'bad' : 'ok', !r ? 'MOTORS' : errs ? `MOTORS ${errs} ERR` : 'MOTORS 8/8');
  const tq = r && r.torque.length ? r.torque.filter(Boolean).length : 0;
  set('#pTorque', !r ? '' : tq === 8 ? 'ok' : 'warn', !r ? 'TORQUE' : tq === 8 ? 'TORQUE ON' : tq ? `TORQUE ${tq}/8` : 'TORQUE OFF');
}
const JN = ['base', 'shoulder', 'elbow', 'wrist flex', 'forearm roll', 'wrist pitch', 'wrist roll'];
function hud(r) {
  let h = '';
  r.deg.forEach((d, i) => { h += `<div class="row"><span>${JN[i]}</span><span class="bar"><b style="width:${(100 * r.norm[i]).toFixed(0)}%"></b></span><span>${fmt(d, 0)}°</span></div>`; });
  h += `<div class="row"><span>gripper</span><span class="bar"><b style="width:${r.gripper}%;background:var(--accent)"></b></span><span>${r.gripper}%</span></div>`;
  h += `<div class="tip">fingertips ${fmt(r.tip[0])} ${fmt(r.tip[1])} ${fmt(r.tip[2])} cm</div>`;
  $('#hud').innerHTML = h;
}

function W(name) { return S.scene && S.scene.world ? S.scene.world.find((w) => w.known && w.label === name) : null; }
function objXY(name) {  // from the server's world model: smoothed, gripper-camera measurement kept until moved
  const w = W(name);
  if (!w) return null;
  const how = w.src === 'gripper' ? 'measured by the gripper camera' : w.on ? `on the ${w.on} · table camera` : 'table camera';
  return { xy: [w.x, w.y], z: w.z || 0, how, age: w.age };
}
function setTarget(o, x, y, z) { if (o.target.distanceTo(new THREE.Vector3(x, y, z)) > 0.05 || !o.g.visible) o.target.set(x, y, z); }

// unknown things on the mat (named by Claude): translucent footprint boxes
const UNK = {};
const MAT_UNK = new THREE.MeshStandardMaterial({ color: 0x7aa2ff, transparent: true, opacity: 0.28, roughness: 0.6 });
function updateUnknowns(world) {
  const live = new Set();
  for (const w of world.filter((w) => !w.known && w.confirmed)) {
    live.add(w.id);
    let u = UNK[w.id];
    if (!u) {
      const g = new THREE.Group();
      const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), MAT_UNK);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0x9db8ff }));
      const l = label('', 'lbl unk');
      g.add(box, edges, l);
      scene.add(g);
      u = UNK[w.id] = { g, box, edges, l, target: new THREE.Vector3(w.x, w.y, 0) };
      g.position.copy(u.target);
    }
    const vessel = /mug|cup|glass|bowl|jar|vase/.test(w.name || '');
    if (vessel && !u.glass) {  // a cup / glass named by Claude: draw a glass cylinder instead of a box
      u.box.visible = u.edges.visible = false;
      u.glass = new THREE.Mesh(new THREE.CylinderGeometry(1, 0.9, 1, 40, 1, true),
        new THREE.MeshStandardMaterial({ color: 0xcfe8ff, transparent: true, opacity: 0.32, roughness: 0.05, metalness: 0.1, side: THREE.DoubleSide }));
      u.glass.rotation.x = Math.PI / 2;
      u.g.add(u.glass);
    }
    const h = vessel ? 9.5 : 2.5;
    if (u.glass) { const r = Math.max(Math.min(w.w_cm, w.d_cm, 12) / 2, 3); u.glass.scale.set(r, h, r); u.glass.position.z = h / 2; }
    else for (const m of [u.box, u.edges]) { m.scale.set(Math.max(w.w_cm, 1.5), Math.max(w.d_cm, 1.5), h); m.position.z = h / 2; }
    u.l.position.set(0, 0, h + 3);
    u.target.set(w.x, w.y, 0);
    u.l.element.innerHTML = (w.name ? esc(w.name) : 'new object') +
      `<small>${w.by_user ? 'named by you' : w.src === 'claude' ? 'seen by Claude' : w.name ? "Claude's guess · click to correct" : w.naming ? 'Claude is looking at it…' : 'unknown, waiting for Claude'} · (${fmt(w.x)}, ${w.y.toFixed(1)}) cm</small>`;
    u.l.element.dataset.id = w.id;
  }
  for (const id of Object.keys(UNK)) if (!live.has(id)) { scene.remove(UNK[id].g); delete UNK[id]; }
}

function updateObjects() {
  const sc = S.scene, st = sc.story || {};
  updateUnknowns(sc.world || []);
  for (const [name, o] of Object.entries(OBJ)) {
    const d = objXY(name);
    const seenNow = d && d.age < 1.5;
    let state = 'unseen', text = '';
    const mugName = st.release && (st.release.label || 'white mug');
    const mug = mugName ? objXY(mugName) : null;
    const bounced = seenNow && mug && Math.hypot(d.xy[0] - mug.xy[0], d.xy[1] - mug.xy[1]) > 6;  // on the mat after the drop
    // the camera wins: a ball the table camera sees on the mat, away from the fingertips, is not in the gripper
    const awayFromTip = seenNow && tip.p1 && Math.hypot(d.xy[0] - tip.cur.x, d.xy[1] - tip.cur.y) > 6;
    if (name === 'orange ball' && st.held && !awayFromTip) { state = 'held'; text = 'held in the gripper'; }
    else if (name === 'orange ball' && st.release && mug && !bounced && !(st.t_end && seenNow) && st.verdict !== 'MISSED') {
      // released over the mug: it is in there (or falling in) unless a camera sees it somewhere else
      setTarget(o, mug.xy[0], mug.xy[1], 0.3); state = 'inmug';
      text = (st.check && st.check.verdict === 'in') || st.verdict === 'SUCCESS' ? `inside the ${mugName} ✓ (seen from above)` : `dropped into the ${mugName}…`;
    } else if (d) {
      setTarget(o, d.xy[0], d.xy[1], d.z);
      state = d.age < 4 ? 'seen' : 'ghost';
      text = state === 'seen' ? `(${fmt(d.xy[0])}, ${d.xy[1].toFixed(1)}) cm · ${d.how}` : `last seen ${d.age.toFixed(0)} s ago`;
    }
    if (state !== 'unseen' && !o.g.visible) { o.g.visible = true; o.g.position.copy(o.target); }
    if (state === 'unseen') o.g.visible = false;
    o.state = state;
    setOpacity(o, state === 'ghost' ? 0.28 : 1);
    const isTarget = (sc.active && sc.active === name) || (st.target && st.target.label === name && ['pick'].includes(st.phase));
    o.l.element.className = 'lbl' + (isTarget ? ' target' : '') + (state === 'ghost' ? ' ghost' : '');
    o.l.element.innerHTML = `${name}<small>${text}</small>`;
    if (name.includes('mug') && d) o.g.rotation.z = Math.atan2(d.xy[1], d.xy[0]);  // handle away from the base (real direction unknown)
  }
  // fusion markers: table camera estimate -> gripper camera measurement (kept until the object is moved)
  const m = (sc.world || []).find((w) => w.known && w.src === 'gripper' && w.table_said);
  if (m) {
    const [tx, ty] = m.table_said, mx = m.x, my = m.y;
    fTable.position.set(tx, ty, 0.15); fMeas.position.set(mx, my, 0.16);
    fGeo.setFromPoints([new THREE.Vector3(tx, ty, 0.15), new THREE.Vector3(mx, my, 0.15)]);
    fLblT.position.set(tx, ty, 0.5); fLblT.element.innerHTML = 'table cam said<small>(' + fmt(tx) + ', ' + ty.toFixed(1) + ')</small>';
    fLblM.position.set(mx, my, 1.5);
    fLblM.element.innerHTML = `gripper cam measured<small>Δ ${Math.hypot(mx - tx, my - ty).toFixed(1)} cm correction</small>`;
    fuse.visible = true;
    fLblT.visible = fLblM.visible = !['drop', 'verify'].includes(st.phase);
  } else fuse.visible = false;
  if (st.release && st.release.tip && !st.t_end) {
    const [rx, ry] = st.release.tip, [gx, gy] = st.release.target || st.release.tip;
    relX.position.set(rx, ry, 0.2);
    relL.position.set(rx, ry, -0.5);
    relL.element.innerHTML = `released here<small>${Math.hypot(rx - gx, ry - gy).toFixed(1)} cm from the measured centre</small>`;
    relMark.visible = true;
    relL.visible = ['drop', 'verify'].includes(st.phase);
  } else relMark.visible = false;
}

// ------------------------------------------------------------------------------------------- story, pipeline, stream
const PH = [['locate', 'LOCATE'], ['measure', 'MEASURE'], ['pick', 'PICK'], ['carry', 'CARRY'], ['drop', 'DROP'], ['verify', 'VERIFY'], ['done', 'DONE']];
function story() {
  const st = S.scene.story;
  $('#infoLine').textContent = S.scene.info || '';
  $('#taskName').textContent = st.task || 'idle';
  $('#taskName').style.color = st.task ? 'var(--text)' : 'var(--muted)';
  $('#phaseChip').textContent = st.phase;
  const v = $('#verdict');
  v.className = st.verdict || ''; v.textContent = st.verdict || '';
  const nowS = Date.now() / 1000;
  let h = '';
  const idx = PH.findIndex(([k]) => k === st.phase);
  PH.forEach(([k, name], i) => {
    const t = st.phase_t[k];
    const next = PH.slice(i + 1).map(([kk]) => st.phase_t[kk]).find(Boolean);
    let cls = '', span = '–';
    if (t != null) {
      const end = next || (st.t_end && k !== 'done' ? st.t_end : null) || (k === 'done' ? t : nowS);
      span = k === 'done' ? (st.seconds ? st.seconds.toFixed(0) + ' s' : '✓') : ((end - t).toFixed(1) + ' s');
      cls = i < idx || st.phase === 'done' ? 'done' : 'active';
    }
    h += `<div class="step ${cls}"><b>${name}</b><span>${span}</span></div>`;
  });
  $('#pipe').innerHTML = h;
  timerBase = st.t0 ? { t0: st.t0, end: st.t_end } : null;
  facts(st);
}
let timerBase = null;
function tickTimer() {
  if (!timerBase) { $('#timer').textContent = '00:00.0'; return; }
  const s = Math.max(0, (timerBase.end || Date.now() / 1000) - timerBase.t0);
  $('#timer').textContent = String(Math.floor(s / 60)).padStart(2, '0') + ':' + (s % 60).toFixed(1).padStart(4, '0');
}
function facts(st) {
  const f = [];
  if (st.measured) {
    const [tx, ty] = st.measured.table, [mx, my] = st.measured.measured;
    f.push(['Mug: table cam → gripper cam correction', Math.hypot(mx - tx, my - ty).toFixed(1) + ' cm']);
  }
  if (st.target) f.push([`Target (${st.target.label})`, `(${fmt(st.target.xy[0])}, ${st.target.xy[1].toFixed(1)}) cm`]);
  if (st.grasped && st.grasped.tip) f.push(['Grasped at', st.grasped.tip.map((v) => v.toFixed(1)).join(', ') + ' cm']);
  if (st.release && st.release.tip && st.release.target)
    f.push(['Release error', Math.hypot(st.release.tip[0] - st.release.target[0], st.release.tip[1] - st.release.target[1]).toFixed(1) + ' cm']);
  if (st.check) f.push(['Check from above', st.check.verdict === 'in' ? 'ball inside the rim' : st.check.verdict === 'out' ? 'ball outside' : 'unclear']);
  if (st.seconds) f.push(['Total time', st.seconds.toFixed(0) + ' s']);
  $('#facts').innerHTML = f.length ? f.map(([a, b]) => `<div class="fact"><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join('')
    : '<div class="fact"><span>No task yet</span><b>–</b></div>';
}

let lastT = 0, typing = null;
const stream = $('#stream');
const ICON = { task: 'TASK', detect: 'SEE', think: 'THINK', move: 'MOVE', gripper: 'GRIP', result: 'RESULT', error: 'ERROR', claude: 'CLAUDE' };
async function pollEvents() {
  try {
    const { events } = await api('/api/events?since=' + lastT);
    if (lastT === 0 && events.length) stream.innerHTML = '';
    for (const e of events) { addEvent(e); lastT = Math.max(lastT, e.t); }
    if (lastT === 0 && !events.length) lastT = 1e-6;
  } catch (e) { /* retry */ }
  setTimeout(pollEvents, 500);
}
function addEvent(e) {
  const st = S.scene && S.scene.story;
  const rel = st && st.t0 && e.t >= st.t0 - 0.5 ? '+' + (e.t - st.t0).toFixed(1) + 's' : new Date(e.t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  const bad = /\b(MISSED|FAILED|ABORT|REFUSED)/.test(e.text);
  const d = document.createElement('div');
  d.className = 'ev ' + e.kind;
  d.innerHTML = `<div class="t">${rel}</div><div class="k ${e.kind}${bad && e.kind === 'result' ? ' bad' : ''}">${ICON[e.kind] || e.kind.toUpperCase()}</div>` +
    `<div class="x">${esc(e.text)}</div><div>${e.img ? `<img src="/img/${e.img}" alt="">` : ''}</div>`;
  stream.prepend(d);
  while (stream.children.length > 140) stream.lastChild.remove();
  if (['think', 'claude', 'result', 'detect', 'task', 'gripper', 'error'].includes(e.kind)) typeNow(e);
}
function typeNow(e) {
  $('#nowKind').textContent = { think: 'thinking', claude: 'Claude is calling a tool', result: 'result', detect: 'seeing',
                                task: 'task', gripper: 'gripper decision', error: 'problem' }[e.kind] || e.kind;
  const el = $('#nowText'), txt = e.text;
  clearInterval(typing);
  let i = 0;
  typing = setInterval(() => {
    i = Math.min(txt.length, i + 3);
    el.innerHTML = esc(txt.slice(0, i)) + '<span class="caret"></span>';
    if (i >= txt.length) clearInterval(typing);
  }, 16);
}

async function pollStats() {
  try {
    const s = await api('/api/stats');
    const td = s.today > 0, n = td ? s.today : s.runs, k = td ? s.today_success : s.success, when = td ? 'today' : 'overall';
    const rate = n ? Math.round(100 * k / n) : null;
    $('#stats').innerHTML = [
      [k + ' / ' + n, 'successful runs ' + when], [rate == null ? '–' : rate + '%', 'success rate ' + when],
      [s.avg_s == null ? '–' : s.avg_s.toFixed(0) + ' s', 'avg time (successes)'], [s.best_s == null ? '–' : s.best_s.toFixed(0) + ' s', 'best time'],
    ].map(([b, t]) => `<div class="stat"><b>${b}</b><span>${t}</span></div>`).join('');
    $('#statNote').textContent = s.note || `${s.success} / ${s.runs} runs succeeded overall`;
    $('#runs').innerHTML = s.recent.map((r) => `<div class="run"><span class="tm">${new Date(r.t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })}</span>` +
      `<span>${esc(r.task.replace('put the orange ball', 'ball'))}</span><span class="vchip ${r.verdict}"${r.why ? ` title="corrected after checking the images (auto check said ${esc(r.auto)}): ${esc(r.why)}"` : ''}>${r.verdict}${r.why ? ' ✎' : ''}</span><span class="s">${r.seconds ? r.seconds.toFixed(0) + ' s' : ''}</span></div>`).join('')
      || '<div class="run"><span></span><span>No runs yet</span></div>';
  } catch (e) { /* retry */ }
  setTimeout(pollStats, 4000);
}

// ------------------------------------------------------------------------------------------- cameras, buttons, zoom
let camN = 0;
function pollCams() {
  camN++;
  $('#camT').src = '/live/table.jpg?' + camN;
  $('#camG').src = '/live/gripper.jpg?' + camN;
  setTimeout(pollCams, 700);
}
document.addEventListener('click', (ev) => {
  const t = ev.target;
  const unk = t.closest && t.closest('.lbl.unk');
  if (unk) {
    const name = prompt('What is this object?', unk.firstChild ? unk.firstChild.textContent : '');
    if (name !== null) fetch('/api/world/rename', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: unk.dataset.id, name }) });
    return;
  }
  if (t.tagName === 'IMG' && (t.closest('.ev') || t.closest('.cams'))) { $('#zoom img').src = t.src; $('#zoom').style.display = 'flex'; }
  else if (t.closest('#zoom')) $('#zoom').style.display = 'none';
});
document.querySelectorAll('#viewBtns [data-v]').forEach((b) => b.addEventListener('click', () => { follow = false; $('#bFollow').classList.remove('on'); setView(b.dataset.v); }));
$('#bFollow').addEventListener('click', () => { follow = !follow; $('#bFollow').classList.toggle('on', follow); });
$('#bSpin').addEventListener('click', () => { controls.autoRotate = !controls.autoRotate; $('#bSpin').classList.toggle('on', controls.autoRotate); });
$('#bClear').addEventListener('click', () => fetch('/clear', { method: 'POST' }));
controls.autoRotateSpeed = 0.6;

// ------------------------------------------------------------------------------------------- render loop
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false); labels.setSize(w, h);
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
resize();
const clock = new THREE.Clock();
function frame() {
  requestAnimationFrame(frame);
  const now = performance.now(), dt = Math.min(clock.getDelta(), 0.1);
  for (const L of Object.values(links)) {
    if (!L.p1) continue;
    const a = Math.min(1, (now - L.t0) / LERP_MS);
    L.cp.lerpVectors(L.p0, L.p1, a); L.cq.slerpQuaternions(L.q0, L.q1, a);
    L.mesh.matrix.compose(L.cp, L.cq, ONE); L.mesh.matrixWorldNeedsUpdate = true;
  }
  if (tip.p1) {
    tip.cur.lerpVectors(tip.p0, tip.p1, Math.min(1, (now - tip.t0) / LERP_MS));
    tipDot.position.copy(tip.cur);
    pushTrail(tip.cur);
    const len = tip.axis.z < -0.05 ? Math.min(tip.cur.z / -tip.axis.z, 40) : 25;
    const end = tip.cur.clone().addScaledVector(tip.axis, len);
    rayGeo.setFromPoints([tip.cur, end]); ray.computeLineDistances();
    rayHit.position.set(end.x, end.y, 0.2); rayHit.visible = tip.axis.z < -0.05;
  }
  for (const u of Object.values(UNK)) u.g.position.lerp(u.target, 1 - Math.exp(-dt * 6));
  for (const [name, o] of Object.entries(OBJ)) {
    if (!o.g.visible) continue;
    if (o.state === 'held' && tip.p1) { o.g.position.set(tip.cur.x, tip.cur.y, tip.cur.z - 2.25); continue; }
    o.g.position.lerp(o.target, 1 - Math.exp(-dt * 8));
  }
  // the object being worked on: pulsing ring + dashed line from the fingertips with the live distance
  const sc = S.scene, st = sc && sc.story;
  const tname = sc && (sc.active || (st && st.target && st.phase === 'pick' ? st.target.label : null));
  const T = tname && OBJ[tname] && OBJ[tname].g.visible && OBJ[tname].state !== 'held' ? OBJ[tname] : null;
  ring.visible = toLine.visible = !!T; toLbl.visible = !!T;
  if (T) {
    const p = T.g.position, s = 1 + 0.08 * Math.sin(now / 220);
    ring.position.set(p.x, p.y, 0.2); ring.scale.set(s, s, 1);
    if (tip.p1) {
      const top = new THREE.Vector3(p.x, p.y, tname.includes('mug') ? 10 : tname === 'rubiks cube' ? 5.7 : 2.25);
      toGeo.setFromPoints([tip.cur, top]); toLine.computeLineDistances();
      toLbl.position.copy(tip.cur.clone().lerp(top, 0.3));
      toLbl.element.textContent = tip.cur.distanceTo(top).toFixed(1) + ' cm to ' + tname;
    }
  }
  if (follow && tip.p1) controls.target.lerp(tip.cur, 1 - Math.exp(-dt * 3));
  if (viewAnim) {
    const a = Math.min(1, (now - viewAnim.s) / 900), e = a * a * (3 - 2 * a);
    camera.position.lerpVectors(viewAnim.p0, viewAnim.p1, e); controls.target.lerpVectors(viewAnim.t0, viewAnim.t1, e);
    if (a >= 1) viewAnim = null;
  }
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  tickTimer();
}


// ------------------------------------------------------------------------------------------- overlays toggles
const LBL_MODES = ['full', 'names', 'off'];
const store = { get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };
let lblMode = store.get('lbl', 'names'), hudOn = store.get('hud', 'on') === 'on';
function applyOverlays() {
  view.classList.toggle('names', lblMode === 'names');
  view.classList.toggle('nolabels', lblMode === 'off');
  $('#bLabels').textContent = 'Labels: ' + lblMode;
  $('#bHud').textContent = 'HUD: ' + (hudOn ? 'on' : 'off');
  $('#hud').classList.toggle('hide', !hudOn); $('#legend').classList.toggle('hide', !hudOn);
}
$('#bLabels').addEventListener('click', () => { lblMode = LBL_MODES[(LBL_MODES.indexOf(lblMode) + 1) % 3]; store.set('lbl', lblMode); applyOverlays(); });
$('#bHud').addEventListener('click', () => { hudOn = !hudOn; store.set('hud', hudOn ? 'on' : 'off'); applyOverlays(); });
applyOverlays();

// ------------------------------------------------------------------------------------------- Claude · MCP drawer
const drawer = { open: store.get('drawer', 'off') === 'on' };
function setDrawer(open) { drawer.open = open; document.body.classList.toggle('drawer', open); store.set('drawer', open ? 'on' : 'off'); setTimeout(resize, 300); if (open) $('#chatBadge').style.display = 'none'; }
$('#bClaude').addEventListener('click', () => setDrawer(!drawer.open));
$('#dClose').addEventListener('click', () => setDrawer(false));
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t.id === 'tab-' + b.dataset.tab));
}));
setDrawer(drawer.open);
let chatI = 0, chatBusy = false;
const chatLog = $('#chatLog');
function argStr(a) { return Object.entries(a || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? Math.round(v * 10) / 10 : JSON.stringify(v)}`).join(', '); }
function addMsg(m) {
  const d = document.createElement('div');
  d.className = 'm ' + m.role + (m.error ? ' err' : '');
  d.textContent = m.role === 'tool' ? `→ ${m.text}(${argStr(m.input)})` : m.role === 'done' ? `✓ done in ${m.text}` : m.text;
  chatLog.appendChild(d);
}
async function pollChat() {
  try {
    const c = await api('/api/chat?since=' + chatI);
    $('#chatNote').style.display = c.sim ? 'block' : 'none';
    for (const m of c.msgs) { addMsg(m); chatI = m.i; if (!drawer.open) $('#chatBadge').style.display = 'inline'; }
    let b = chatLog.querySelector('.busy');
    if (c.busy && !b) { b = document.createElement('div'); b.className = 'busy'; b.textContent = 'Claude is working'; chatLog.appendChild(b); }
    if (!c.busy && b) b.remove();
    if (b) chatLog.appendChild(b);
    if (c.msgs.length || c.busy !== chatBusy) chatLog.scrollTop = chatLog.scrollHeight;
    chatBusy = c.busy;
    $('#chatSend').disabled = c.busy; $('#chatModel').value = c.model;
  } catch (e) { /* retry */ }
  setTimeout(pollChat, 600);
}
async function post(path, body) { return (await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })).json(); }
async function sendChat(text) {
  text = text.trim();
  if (!text) return;
  $('#chatIn').value = '';
  const r = await post('/api/chat', { text });
  if (!r.ok) addMsg({ role: 'error', text: r.error });
}
$('#chatForm').addEventListener('submit', (e) => { e.preventDefault(); sendChat($('#chatIn').value); });
$('#chatIn').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat($('#chatIn').value); } });
document.querySelectorAll('#chips button').forEach((b) => b.addEventListener('click', () => sendChat(b.textContent)));
$('#chatStop').addEventListener('click', () => post('/api/chat/stop'));
$('#chatNew').addEventListener('click', () => post('/api/chat/new'));
$('#chatModel').addEventListener('change', (e) => post('/api/chat/model', { model: e.target.value }));
async function loadMcp() {
  try {
    const m = await api('/api/mcp');
    $('#mcpCfg').textContent = JSON.stringify(m.config, null, 2);
    $('#mcpInstr').textContent = m.instructions;
    document.querySelector('.tabs [data-tab=tools]').textContent = `Tools (${m.tools.length})`;
    $('#toolList').innerHTML = m.tools.map((t) => {
      const props = Object.entries((t.inputSchema && t.inputSchema.properties) || {});
      return `<div class="toolc"><b>${esc(t.name)}</b><p>${esc(t.description)}</p>` +
        (props.length ? `<small>${props.map(([k, v]) => esc(k) + ': ' + esc(v.type || '')).join(' · ')}</small>` : '<small>no parameters</small>') + '</div>';
    }).join('');
  } catch (e) { /* retry on next open */ }
}

// ------------------------------------------------------------------------------------------- Ask Claude what's here
async function pollScan(wasBusy) {
  try {
    const r = await api('/api/scan');
    const b = $('#bScan');
    if (r.busy) { b.textContent = '◆ Claude is looking…'; b.disabled = true; setTimeout(() => pollScan(true), 1000); return; }
    b.disabled = false;
    b.textContent = wasBusy ? (r.error ? '◆ scan failed: ' + r.error.slice(0, 40) : `◆ Claude found ${r.found.length} more`) : "◆ Ask Claude what's here";
    if (wasBusy) setTimeout(() => { b.textContent = "◆ Ask Claude what's here"; }, 6000);
  } catch (e) { setTimeout(() => pollScan(wasBusy), 1500); }
}
$('#bScan').addEventListener('click', async () => { await fetch('/api/scan', { method: 'POST' }); pollScan(true); });

// ------------------------------------------------------------------------------------------- auto-reload
let version = null;
async function pollVersion() {
  try { const { v } = await api('/api/version'); if (version === null) version = v; else if (v !== version) location.reload(); } catch (e) { /* server restarting */ }
  setTimeout(pollVersion, 1500);
}

// bookmarkable views: ?v=orbit|top|table|side  &follow=1  &spin=1
const QS = new URLSearchParams(location.search);
if (VIEWS[QS.get('v')]) { camera.position.set(...VIEWS[QS.get('v')][0]); controls.target.set(...VIEWS[QS.get('v')][1]); controls.update(); }
if (QS.get('follow') === '1') { follow = true; $('#bFollow').classList.add('on'); }
if (QS.get('spin') === '1') { controls.autoRotate = true; $('#bSpin').classList.add('on'); }
if (QS.get('claude') === '1') setDrawer(true);
loadArm().then(() => { pollRobot(); });
pollScene(); pollEvents(); pollStats(); pollCams(); pollChat(); loadMcp(); pollVersion(); frame();
