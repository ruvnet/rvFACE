/**
 * 1:1 compare pane. The left slot is "Compare with" — it starts on your live
 * **webcam** but can switch to an **Image**; the right slot is a fixed
 * **reference image**. Switching a running webcam to Image freezes its last
 * (clean, overlay-free) frame as the image, so you can compare a snapshot
 * without uploading — an explicit Upload button always lets you pick a file
 * instead. When both slots hold a face the similarity (0–100) is shown as a
 * gauge with the upstream threshold-75 verdict, updating live while the camera
 * runs. Each face is also tagged with a heuristic expression guess.
 */

import { MATCH_THRESHOLD, type FaceResult, type RvFaceEngine } from '../engine';
import { drawFaces } from './overlay';
import { makeDropZone, type DropZoneHandle } from './dropzone';
import { decodeImageFile, FrameGrabber } from './frame';
import { estimateExpression } from './expression';
import { Webcam } from './webcam';
import type { StatusBar } from './statusbar';

type Source = 'image' | 'webcam';

interface SlotState {
  face: FaceResult | null;
  latencyMs: number;
}

/** A still image held by a slot — uploaded, or frozen from the webcam. */
interface LoadedImage {
  bitmap: ImageBitmap;
  face: FaceResult | null;
  latencyMs: number;
  width: number;
  height: number;
  origin: 'upload' | 'webcam';
}

interface SlotOptions {
  /** Internal id used in logs and element names (e.g. "A"). */
  label: string;
  /** Human headline shown above the slot (e.g. "Reference image"). */
  headline: string;
  /** Whether this slot offers a live-webcam source in addition to an image. */
  allowWebcam: boolean;
  /** Source selected on load; a `webcam` default auto-starts once the engine is ready. */
  defaultSource: Source;
}

class CompareSlot {
  readonly state: SlotState = { face: null, latencyMs: 0 };
  private readonly label: string;
  private readonly zone: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly video: HTMLVideoElement;
  private readonly hint: HTMLElement;
  private readonly info: HTMLElement;
  private readonly radios: HTMLInputElement[];
  private readonly grabber = new FrameGrabber();
  private readonly webcam: Webcam;
  private readonly dropZone: DropZoneHandle;
  private image: LoadedImage | null = null;
  private pendingAutoStart = false;
  private lastFrameTime = 0;
  private fpsEma = 0;

  constructor(
    root: HTMLElement,
    opts: SlotOptions,
    private readonly getEngine: () => RvFaceEngine | null,
    private readonly status: StatusBar,
    private readonly onChange: () => void,
  ) {
    this.label = opts.label;
    const lc = opts.label.toLowerCase();
    const camDefault = opts.allowWebcam && opts.defaultSource === 'webcam';
    const toggle = opts.allowWebcam
      ? `
          <div class="source-toggle" role="radiogroup" aria-labelledby="src-${lc}-label">
            <label class="seg">
              <input type="radio" name="src-${lc}" value="image"${camDefault ? '' : ' checked'}>
              <span>Image</span>
            </label>
            <label class="seg">
              <input type="radio" name="src-${lc}" value="webcam"${camDefault ? ' checked' : ''}>
              <span>Webcam</span>
            </label>
          </div>`
      : '';
    root.className = 'slot';
    root.innerHTML = `
      <div class="slot-head">
        <span class="slot-label" id="src-${lc}-label">${opts.headline}</span>
        <div class="slot-controls">
          ${toggle}
          <button type="button" class="slot-upload" data-role="upload">Upload image</button>
        </div>
      </div>
      <div class="drop-zone drop-zone-slot" data-role="zone">
        <p class="dz-hint" data-role="hint">
          <span class="dz-cta">Click to upload an image</span>
          <span class="dz-sub">or drop a file here</span>
        </p>
        <canvas data-role="view" role="img" aria-label="${opts.headline} preview" hidden></canvas>
        <video data-role="video" playsinline muted hidden></video>
      </div>
      <div class="face-info mono" data-role="info">No image yet.</div>
    `;
    this.zone = root.querySelector('[data-role="zone"]')!;
    this.canvas = root.querySelector('[data-role="view"]')!;
    this.ctx = this.canvas.getContext('2d')!;
    this.video = root.querySelector('[data-role="video"]')!;
    this.hint = root.querySelector('[data-role="hint"]')!;
    this.info = root.querySelector('[data-role="info"]')!;
    this.radios = [...root.querySelectorAll<HTMLInputElement>(`input[name="src-${lc}"]`)];
    this.webcam = new Webcam(this.video, (video) => this.onCamFrame(video));
    this.pendingAutoStart = camDefault;

    this.dropZone = makeDropZone(this.zone, (file) => void this.load(file), {
      ariaLabel: `${opts.headline}: drop a file or activate to browse`,
    });
    const upload = root.querySelector<HTMLButtonElement>('[data-role="upload"]')!;
    upload.setAttribute('aria-label', `Upload an image for ${opts.headline}`);
    upload.addEventListener('click', () => this.dropZone.open());
    for (const radio of this.radios) {
      radio.addEventListener('change', () => {
        if (radio.checked) void this.selectSource(radio.value as Source);
      });
    }
  }

  /** Auto-start a webcam-default slot the first time the engine is ready. */
  notifyEngineReady(): void {
    if (this.pendingAutoStart && !this.webcam.active) {
      this.pendingAutoStart = false;
      void this.startWebcam();
    }
  }

  /** React to the source toggle (webcam slots only). */
  private async selectSource(src: Source): Promise<void> {
    if (src === 'webcam') {
      await this.startWebcam();
    } else {
      await this.switchToImage();
    }
  }

  private setSourceUI(src: Source): void {
    for (const radio of this.radios) radio.checked = radio.value === src;
  }

  /**
   * Leave webcam mode: freeze the last clean frame as the slot's image so the
   * comparison can continue without an upload. Falls back to whatever image
   * was already loaded (or the empty prompt) if no frame is available.
   */
  private async switchToImage(): Promise<void> {
    if (this.webcam.active) {
      const frozen = await this.captureFrozenFrame();
      this.stopWebcam();
      if (frozen) {
        this.image?.bitmap.close();
        this.image = frozen;
        this.status.log(`compare ${this.label}: froze webcam frame (use it, or upload an image)`);
      }
    }
    this.showImage();
  }

  /** Grab the current webcam frame (no overlay) and analyze it, for freezing. */
  private async captureFrozenFrame(): Promise<LoadedImage | null> {
    const engine = this.getEngine();
    if (!engine || this.video.videoWidth === 0) return null;
    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(this.video); // clean pixels, no drawn boxes
    } catch {
      return null;
    }
    const frame = this.grabber.grab(bitmap, bitmap.width, bitmap.height);
    const t0 = performance.now();
    const faces = await engine.analyze(frame.rgba, frame.width, frame.height, 1);
    return {
      bitmap,
      face: faces[0] ?? null,
      latencyMs: performance.now() - t0,
      width: frame.width,
      height: frame.height,
      origin: 'webcam',
    };
  }

  private async load(file: File): Promise<void> {
    const engine = this.getEngine();
    if (!engine) {
      this.status.log('engine not ready yet', 'warn');
      return;
    }
    // Uploading always switches this slot to an image.
    if (this.webcam.active) this.stopWebcam();
    this.setSourceUI('image');

    let bitmap: ImageBitmap;
    try {
      bitmap = await decodeImageFile(file);
    } catch (err) {
      this.status.log(String(err instanceof Error ? err.message : err), 'error');
      return;
    }

    const frame = this.grabber.grab(bitmap, bitmap.width, bitmap.height);
    const t0 = performance.now();
    const faces = await engine.analyze(frame.rgba, frame.width, frame.height, 1);
    const latencyMs = performance.now() - t0;

    this.image?.bitmap.close();
    this.image = {
      bitmap, face: faces[0] ?? null, latencyMs, width: frame.width, height: frame.height, origin: 'upload',
    };
    this.showImage();

    if (this.state.face) {
      this.status.log(`compare image ${this.label}: face ready (${latencyMs.toFixed(1)} ms)`);
    } else {
      this.status.log(`compare image ${this.label}: no face found`, 'warn');
    }
  }

  /** Render the stored image (or the empty hint) and publish its face. */
  private showImage(): void {
    if (this.image) {
      this.showFrame(this.image.width, this.image.height);
      this.ctx.drawImage(this.image.bitmap, 0, 0, this.image.width, this.image.height);
      // A frozen webcam still is shown clean (no boxes/landmarks); uploaded
      // images keep the detection overlay. The face is still used for scoring.
      if (this.image.origin !== 'webcam') {
        drawFaces(this.ctx, this.image.face ? [this.image.face] : []);
      }
      this.state.face = this.image.face;
      this.state.latencyMs = this.image.latencyMs;
    } else {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.canvas.hidden = true;
      this.hint.hidden = false;
      this.state.face = null;
      this.state.latencyMs = 0;
    }
    this.renderInfo(this.image ? (this.image.origin === 'webcam' ? 'frozen' : 'image') : 'empty');
    this.onChange();
  }

  private async startWebcam(): Promise<void> {
    if (!this.getEngine()) {
      this.status.log('engine not ready yet', 'warn');
      this.setSourceUI('image');
      this.showImage();
      return;
    }
    this.zone.setAttribute('aria-disabled', 'true');
    this.hint.hidden = true;
    try {
      await this.webcam.start();
    } catch (err) {
      this.status.log(`webcam unavailable: ${err instanceof Error ? err.message : err}`, 'error');
      this.status.log(`compare ${this.label}: use an image instead — click "Upload image" or drop a file`);
      this.stopWebcam();
      this.showImage();
      return;
    }
    this.fpsEma = 0;
    this.lastFrameTime = performance.now();
    this.status.log(`compare webcam ${this.label}: started`);
  }

  /** Stop the camera, clear its metrics, and reset the toggle to Image. */
  private stopWebcam(): void {
    if (this.webcam.active) this.webcam.stop();
    this.zone.removeAttribute('aria-disabled');
    this.setSourceUI('image');
    this.status.setFrameMetrics(null);
  }

  /** One webcam iteration: grab -> analyze the primary face -> redraw -> rescore. */
  private async onCamFrame(video: HTMLVideoElement): Promise<void> {
    const engine = this.getEngine();
    if (!engine) return;

    const frame = this.grabber.grab(video, video.videoWidth, video.videoHeight);
    const t0 = performance.now();
    let faces: FaceResult[] = [];
    try {
      faces = await engine.analyze(frame.rgba, frame.width, frame.height, 1);
    } catch (err) {
      this.status.log(
        `compare webcam ${this.label}: analyze failed: ${err instanceof Error ? err.message : err}`,
        'error',
      );
      this.stopWebcam();
      this.showImage();
      return;
    }
    this.state.latencyMs = performance.now() - t0;
    this.state.face = faces[0] ?? null;

    if (!this.webcam.active) return; // stopped while awaiting
    this.showFrame(frame.width, frame.height);
    this.ctx.drawImage(video, 0, 0, frame.width, frame.height);
    drawFaces(this.ctx, faces);
    this.renderInfo('webcam');

    const now = performance.now();
    const dt = now - this.lastFrameTime;
    this.lastFrameTime = now;
    const fps = 1000 / Math.max(dt, 1);
    this.fpsEma = this.fpsEma === 0 ? fps : this.fpsEma * 0.9 + fps * 0.1;
    this.status.setFrameMetrics({ latencyMs: this.state.latencyMs, fps: this.fpsEma });

    this.onChange();
  }

  private showFrame(w: number, h: number): void {
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this.canvas.hidden = false;
    this.hint.hidden = true;
  }

  private renderInfo(kind: 'image' | 'frozen' | 'webcam' | 'empty'): void {
    const face = this.state.face;
    const source =
      kind === 'webcam' ? 'webcam' : kind === 'frozen' ? 'webcam still' : 'image';
    let text: string;
    if (face) {
      const [x1, y1, x2, y2] = face.box;
      const expr = estimateExpression(face.landmarks);
      text =
        `${source}: ${expr.label} · box [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}] · ` +
        `${this.state.latencyMs.toFixed(1)} ms`;
    } else if (kind === 'webcam') {
      text = `webcam: no face in view · ${this.state.latencyMs.toFixed(1)} ms`;
    } else if (kind === 'frozen') {
      text = `webcam still: no face found · ${this.state.latencyMs.toFixed(1)} ms`;
    } else if (kind === 'image') {
      text = `image: no face found · ${this.state.latencyMs.toFixed(1)} ms`;
    } else {
      text = 'No image yet.';
    }
    this.info.textContent = text;
    this.canvas.setAttribute('aria-label', text);
  }
}

export class ComparePane {
  private readonly slotWebcam: CompareSlot;
  private readonly slotReference: CompareSlot;
  private readonly gaugeArc: SVGPathElement;
  private readonly gaugeScore: HTMLElement;
  private readonly verdict: HTMLElement;
  private readonly srStatus: HTMLElement;

  constructor(
    root: HTMLElement,
    private readonly getEngine: () => RvFaceEngine | null,
    status: StatusBar,
  ) {
    root.innerHTML = `
      <div class="pane-head"><h2>1:1 Compare</h2></div>
      <p class="pane-lede">Point your webcam (left) — or choose an image — and compare it against a reference photo (right).</p>
      <div class="compare-grid">
        <div data-role="slot-left"></div>
        <div class="gauge-wrap" role="group" aria-label="Similarity result">
          <svg viewBox="0 0 100 60" class="gauge" aria-hidden="true">
            <path class="gauge-track" d="M 10 55 A 40 40 0 0 1 90 55" />
            <path class="gauge-fill" data-role="arc" d="M 10 55 A 40 40 0 0 1 90 55"
                  pathLength="100" stroke-dasharray="0 100" />
          </svg>
          <div class="gauge-score mono" data-role="score" aria-hidden="true">—</div>
          <div class="verdict" data-role="verdict" aria-hidden="true">awaiting both faces</div>
          <div class="gauge-threshold mono" aria-hidden="true">match &gt; ${MATCH_THRESHOLD}</div>
          <p class="sr-only" role="status" aria-live="polite" data-role="sr">Awaiting both faces.</p>
        </div>
        <div data-role="slot-right"></div>
      </div>
    `;
    const update = () => this.update();
    this.slotWebcam = new CompareSlot(
      root.querySelector('[data-role="slot-left"]')!,
      { label: 'A', headline: 'Compare with', allowWebcam: true, defaultSource: 'webcam' },
      getEngine, status, update,
    );
    this.slotReference = new CompareSlot(
      root.querySelector('[data-role="slot-right"]')!,
      { label: 'B', headline: 'Reference image', allowWebcam: false, defaultSource: 'image' },
      getEngine, status, update,
    );
    this.gaugeArc = root.querySelector('[data-role="arc"]')!;
    this.gaugeScore = root.querySelector('[data-role="score"]')!;
    this.verdict = root.querySelector('[data-role="verdict"]')!;
    this.srStatus = root.querySelector('[data-role="sr"]')!;
  }

  /** Auto-start the webcam slot once the engine has initialized. */
  notifyEngineReady(): void {
    this.slotWebcam.notifyEngineReady();
    this.slotReference.notifyEngineReady();
  }

  /** Recompute the gauge (also called when the engine changes). */
  update(): void {
    const engine = this.getEngine();
    const a = this.slotWebcam.state.face;
    const b = this.slotReference.state.face;
    if (!engine || !a || !b) {
      this.gaugeArc.setAttribute('stroke-dasharray', '0 100');
      this.gaugeArc.classList.remove('gauge-same', 'gauge-diff');
      this.gaugeScore.textContent = '—';
      const msg = a || b ? 'awaiting second face' : 'awaiting both faces';
      this.verdict.textContent = msg;
      this.verdict.className = 'verdict';
      this.srStatus.textContent = `${msg}.`;
      return;
    }
    const score = engine.similarity(a.embedding, b.embedding);
    const same = score > MATCH_THRESHOLD;
    this.gaugeArc.setAttribute('stroke-dasharray', `${score.toFixed(2)} 100`);
    this.gaugeArc.classList.toggle('gauge-same', same);
    this.gaugeArc.classList.toggle('gauge-diff', !same);
    this.gaugeScore.textContent = score.toFixed(1);
    this.verdict.textContent = same ? 'SAME PERSON' : 'DIFFERENT';
    this.verdict.className = `verdict ${same ? 'verdict-same' : 'verdict-diff'}`;
    this.srStatus.textContent =
      `Similarity ${score.toFixed(1)} of 100 — ${same ? 'same person' : 'different person'} ` +
      `(match above ${MATCH_THRESHOLD}).`;
  }
}
