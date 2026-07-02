/**
 * Weights panel — shown when the redistributable detector loaded but the
 * landmark + embedder weights are absent (as on the public Pages demo).
 *
 * Those two networks have no upstream LICENSE (ADR-0003 / models/README.md),
 * so this repo never commits or deploys them. The panel explains that and
 * lets the user drop in their own locally-generated safetensors — produced by
 * `python3 tools/fetch_and_convert.py` — to run the full pipeline. When both
 * files are supplied it calls `onReady`, and `main.ts` initializes the engine
 * exactly as it would with an out-of-the-box complete weight set.
 */

import { readSafetensorsFile, type UserWeights, WEIGHT_FILES } from '../weights';
import type { StatusBar } from './statusbar';

interface Slot {
  key: 'landmark' | 'embedder';
  file: string;
  desc: string;
  bytes: Uint8Array | null;
  row: HTMLElement;
  state: HTMLElement;
}

export class WeightsPanel {
  private readonly el: HTMLElement;
  private readonly slots: Slot[];

  constructor(
    private readonly host: HTMLElement,
    private readonly status: StatusBar,
    private readonly onReady: (weights: UserWeights) => void,
  ) {
    this.el = document.createElement('div');
    this.el.className = 'weights-panel';
    this.el.innerHTML = `
      <strong>Detector loaded (MIT) — two weights needed to run recognition</strong>
      <p>
        The face detector (<code>${WEIGHT_FILES.detector}</code>, MIT lineage)
        ships with this demo. The <b>landmark</b> and <b>embedder</b> weights are
        <b>not redistributable</b> — their upstream sources publish no LICENSE
        (see <code>models/README.md</code>), so they are never committed or
        deployed here.
      </p>
      <p>
        Generate them locally, then drop the two <code>.safetensors</code> files
        below (they stay in your browser — nothing is uploaded):
      </p>
      <pre class="weights-cmd"><code>git clone https://github.com/ruvnet/rvFACE
cd rvFACE &amp;&amp; python3 tools/fetch_and_convert.py</code></pre>
      <div class="weights-slots" data-role="slots"></div>
    `;

    const slotsHost = this.el.querySelector<HTMLElement>('[data-role="slots"]')!;
    const specs: { key: 'landmark' | 'embedder'; file: string; desc: string }[] = [
      { key: 'landmark', file: WEIGHT_FILES.landmark, desc: '68-pt landmarks' },
      { key: 'embedder', file: WEIGHT_FILES.embedder, desc: 'face embedding' },
    ];
    this.slots = specs.map(({ key, file, desc }) => {
      const row = document.createElement('label');
      row.className = 'weights-slot';
      row.innerHTML = `
        <span class="ws-title"><code>${file}</code> <span class="ws-desc">${desc}</span></span>
        <span class="ws-state" data-role="state">drop or pick file</span>
        <input type="file" accept=".safetensors" hidden />
      `;
      const input = row.querySelector<HTMLInputElement>('input')!;
      const state = row.querySelector<HTMLElement>('[data-role="state"]')!;
      const slot: Slot = { key, file, desc, bytes: null, row, state };

      input.addEventListener('change', () => {
        const f = input.files?.[0];
        if (f) void this.accept(slot, f);
        input.value = '';
      });
      row.addEventListener('dragover', (e) => {
        e.preventDefault();
        row.classList.add('drag-over');
      });
      row.addEventListener('dragleave', () => row.classList.remove('drag-over'));
      row.addEventListener('drop', (e) => {
        e.preventDefault();
        row.classList.remove('drag-over');
        const f = e.dataTransfer?.files?.[0];
        if (f) void this.accept(slot, f);
      });

      slotsHost.appendChild(row);
      return slot;
    });
  }

  /** Insert the panel just above the analyze/compare panes. */
  mount(): void {
    if (!this.el.isConnected) this.host.querySelector('.panes')!.before(this.el);
  }

  /** Remove the panel (once both weights are in and the engine is starting). */
  unmount(): void {
    this.el.remove();
  }

  private async accept(slot: Slot, file: File): Promise<void> {
    try {
      slot.bytes = await readSafetensorsFile(file, slot.file);
    } catch (err) {
      slot.bytes = null;
      slot.row.classList.remove('ws-ok');
      slot.state.textContent = err instanceof Error ? err.message : String(err);
      slot.state.classList.add('ws-err');
      this.status.log(`${slot.file}: ${err instanceof Error ? err.message : err}`, 'error');
      return;
    }
    slot.row.classList.add('ws-ok');
    slot.state.classList.remove('ws-err');
    slot.state.textContent = `loaded ${file.name} ✓`;
    this.status.log(`${slot.key} weights loaded (${file.name})`);
    this.maybeReady();
  }

  private maybeReady(): void {
    const landmark = this.slots.find((s) => s.key === 'landmark')?.bytes;
    const embedder = this.slots.find((s) => s.key === 'embedder')?.bytes;
    if (landmark && embedder) this.onReady({ landmark, embedder });
  }
}
