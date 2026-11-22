import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const get = (id) => document.getElementById(id);
const viewport = get('viewport');
const frame = get('frame');
const status = get('status');
const home = get('home');
const optionFields = [
  ['alpha', '透明度阈值（/255）', 1, 0, 12, .5],
  ['opacity', '不透明度倍率', 1, 0, 2, .05],
  ['coverage', '高斯覆盖范围（σ）', 3.33, 1, 4, .01],
  ['minRadius', '最小投影半径（px，低于则剔除）', 0, 0, 10, .1],
  ['maxRadius', '最大投影半径（px）', 512, 16, 1024, 16],
  ['preBlur', '预滤波方差（px²）', 0, 0, 2, .05],
  ['blur', '抗混叠滤波方差（px²）', .3, 0, 2, .05],
  ['clip', '投影雅可比裁剪余量', 1.3, 1, 2, .05],
  ['focal', '投影焦距修正', 1, .5, 2, .05],
  ['near', '近裁剪面', .01, .001, 10, .001],
  ['far', '远裁剪面', 200, 1, 100000, 1],
];
for (const [key, label, value, min, max, step] of optionFields) {
  const row = document.createElement('label');
  row.textContent = label;
  const input = document.createElement('input');
  Object.assign(input, { id: 'opt-' + key, type: 'number', value, min, max, step });
  row.append(input);
  get('numeric-options').append(row);
}
function readOptions() {
  if (get('renderer').value === 'native') return {};
  const result = { background: get('opt-background').value, sort: get('opt-sort').value };
  for (const [key] of optionFields) result[key] = Number(get('opt-' + key).value);
  return result;
}

async function run() {
  const response = await fetch('/api/info');
  if (!response.ok) throw new Error('无法读取服务信息');
  const info = await response.json();
  const { comparison } = info;
  const mode = get('mode');
  const resolution = get('resolution');
  let activeMode = 'default';
  if (comparison) {
    document.title = 'Bicycle 稠密化对比 · 3DGS 查看器';
    get('scene').textContent = `Bicycle · ${comparison.steps.toLocaleString('zh-CN')} 步训练 · ${comparison.testViews} 张留出照片均值`;
    get('mode-control').hidden = false;
    viewport.classList.add('with-caption');
    get('compare-captions').hidden = false;
    mode.value = 'compare';
    for (const mode of ['disabled', 'enabled']) {
      const { count, psnr, ssim } = comparison.models[mode];
      get(mode + '-stats').textContent = `${count.toLocaleString('zh-CN')} 个高斯 · PSNR ${psnr.toFixed(2)} dB · SSIM ${ssim.toFixed(3)}`;
    }
    if (info.originalModel) {
      mode.add(new Option('原演示模型', 'default'));
      get('default-stats').textContent = `${info.originalModel.count.toLocaleString('zh-CN')} 个高斯`;
    }
  } else {
    get('scene').textContent = info.checkpoint + ' · ' + info.count.toLocaleString('zh-CN') + ' 个高斯';
  }
  for (let degree = 0; degree <= info.maxSh; degree++) get('sh').add(new Option(degree, degree));
  get('sh').value = info.maxSh;
  if ([...resolution.options].some((option) => option.value === info.initialResolution)) {
    resolution.value = info.initialResolution;
  }

  function updateMode() {
    activeMode = mode.value;
    const paired = activeMode === 'compare';
    viewport.classList.toggle('comparison', paired);
    get('resolution-label').textContent = paired ? '每侧分辨率' : '分辨率';
    for (const key of ['disabled', 'enabled', 'default']) {
      get(key + '-caption').hidden = paired ? key === 'default' : key !== activeMode;
    }
    frame.alt = paired ? 'Bicycle 同步视角对比，左侧关闭稠密化，右侧开启稠密化'
      : `Bicycle · ${mode.selectedOptions[0].textContent}`;
    for (const id of ['inference', 'npu', 'prepare']) get(id).title = paired ? '左右两侧耗时合计' : '当前画面耗时';
    const allowed = get('renderer').value === 'tunable' ? [...resolution.options].map(o => o.value) : info.viewModes[activeMode];
    for (const option of resolution.options) option.disabled = !allowed.includes(option.value);
    if (!allowed.includes(resolution.value) && allowed.length) {
      resolution.value = allowed.includes(info.initialResolution) ? info.initialResolution : allowed[allowed.length - 1];
    }
    resolution.disabled = !allowed.length;
    frame.hidden = true;
    status.textContent = allowed.length ? '正在载入当前视角…' : '当前显存预算不足，请调高服务端预算或切换模型。';
    status.hidden = false;
  }
  if (comparison) updateMode();

  const camera = new THREE.PerspectiveCamera();
  camera.up.fromArray(info.up).normalize();
  camera.position.fromArray(info.position);
  const controls = new OrbitControls(camera, viewport);
  controls.target.fromArray(info.target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed = 0.65;
  controls.minDistance = info.radius * 0.25;
  controls.maxDistance = info.radius * 15;
  controls.update();
  controls.saveState();

  let dirty = true;
  let working = false;
  let requestedAt = -Infinity;
  let renderVersion = 0;
  let frameUrl;
  if (comparison) {
    mode.addEventListener('change', () => {
      renderVersion++;
      dirty = true;
      updateMode();
      for (const id of ['inference', 'npu', 'prepare', 'roundtrip']) writeTime(id, null);
    });
  }
  function optionsChanged() {
    renderVersion++;
    dirty = true;
  }
  get('advanced-options').addEventListener('change', optionsChanged);
  get('renderer').addEventListener('change', () => {
    const tunable = get('renderer').value === 'tunable';
    get('advanced-options').disabled = !tunable;
    get('renderer-note').textContent = tunable
      ? '可调模式在 NPU 上分块计算，可能明显慢于融合算子。建议先用低分辨率。参数同时作用于左右两图。'
      : '原生模式使用算子固定参数，高级设置暂不生效。';
    if (comparison) updateMode();
    optionsChanged();
  });
  get('reset-options').addEventListener('click', () => {
    get('opt-background').value = '#000000';
    get('opt-sort').value = 'depth';
    for (const [key, , value] of optionFields) get('opt-' + key).value = value;
    optionsChanged();
  });
  controls.addEventListener('change', () => { dirty = true; });
  function resetCamera() {
    controls.enableDamping = false;
    controls.update();
    controls.reset();
    controls.enableDamping = true;
  }
  home.addEventListener('click', resetCamera);
  document.addEventListener('keydown', (event) => {
    if (event.key.toLowerCase() === 'r' && !event.ctrlKey && !event.altKey && !event.metaKey && !home.disabled) resetCamera();
  });
  for (const key of ['resolution', 'sh', 'quality', 'fov']) {
    get(key).addEventListener('input', () => {
      renderVersion++;
      dirty = true;
      if (key === 'quality' || key === 'fov') get(key+'-out').textContent = get(key).value + (key === 'fov' ? '°' : '');
    });
  }
  function cameraToWorld() {
    camera.updateMatrixWorld();
    const m = camera.matrixWorld.elements;
    // Three.js uses right/up/back. gauss-splat takes right/down/forward.
    return [m[0], -m[4], -m[8], m[12], m[1], -m[5], -m[9], m[13],
            m[2], -m[6], -m[10], m[14], 0, 0, 0, 1];
  }

  function writeTime(id, value) {
    get(id).textContent = value === null ? '—' : Number(value).toFixed(1) + ' ms';
  }

  async function fetchFrame() {
    const version = renderVersion;
    dirty = false;
    working = true;
    requestedAt = performance.now();
    let nextUrl;
    try {
      const [width, height] = resolution.value.split('x').map(Number);
      camera.fov = Number(get('fov').value);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      const result = await fetch('/api/frame', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ width, height, fov: camera.fov, c2w: cameraToWorld(), mode: activeMode,
                              quality: Number(get('quality').value), sh: Number(get('sh').value),
                              renderer: get('renderer').value, renderOptions: readOptions() }),
      });
      if (result.status === 429) {
        dirty = true;
        requestedAt = performance.now() + 200;
        return;
      }
      if (!result.ok) throw new Error((await result.json()).error);
      nextUrl = URL.createObjectURL(await result.blob());
      if (version !== renderVersion) return;
      const probe = new Image();
      probe.src = nextUrl;
      await probe.decode();
      if (version !== renderVersion) return;
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = nextUrl;
      nextUrl = undefined;
      frame.src = frameUrl;
      frame.hidden = false;
      status.hidden = true;
      home.disabled = false;
      for (const [id, name] of [['inference', 'X-Inference-Ms'], ['npu', 'X-Npu-Ms'], ['prepare', 'X-Prepare-Ms']]) {
        writeTime(id, result.headers.get(name));
      }
      writeTime('roundtrip', performance.now() - requestedAt);
    } catch (error) {
      if (version === renderVersion) {
        status.textContent = error.message;
        status.hidden = false;
      }
    } finally {
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      working = false;
    }
  }
  function animate(timestamp) {
    controls.update();
    if (dirty && !working && !resolution.disabled
        && timestamp - requestedAt >= 1000 / Number(get('fps').value)) void fetchFrame();
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}
run().catch((error) => { status.textContent = '无法连接渲染服务。' + error.message; });
