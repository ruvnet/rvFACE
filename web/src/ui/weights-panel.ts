/**
 * Weights panel — shown when the shipped base (detector MIT + embedder
 * Apache-2.0) loaded but the landmark weights are absent (as on the public
 * Pages demo). Live detection is already running in that state; this panel
 * collects the ONE remaining file.
 *
 * The landmark network has no upstream LICENSE (ADR-0003 / models/README.md),
 * so this repo never commits or deploys it. The panel explains that and lets
 * the user drop in their own locally-generated safetensors — produced by
 * `python3 tools/fetch_and_convert.py` — to unlock landmarks, pose,
 * alignment and 1:1 compare. When the file is supplied it calls `onReady`,
 * and `main.ts` restarts the engine with the full weight set.
 */

import { readSafetensorsFile, type UserWeights, WEIGHT_FILES } from '../weights';
import type { StatusBar } from './statusbar';

export class WeightsPanel {
  private readonly el: HTMLElement;

  constructor(
    private readonly host: HTMLElement,
    private readonly status: StatusBar,
    private readonly onReady: (weights: UserWeights) => void,
  ) {
    this.el = document.createElement('div');
    this.el.className = 'weights-panel';
    this.el.innerHTML = `
      <strong>Live detection is running — one file unlocks the full pipeline</strong>
      <p>
        The face detector (<code>${WEIGHT_FILES.detector}</code>, MIT lineage)
        and the embedder (<code>${WEIGHT_FILES.embedder}</code>, Apache-2.0)
        ship with this demo, so webcam detection boxes work out of the box.
        The <b>landmark</b> weights are <b>not redistributable</b> — their
        upstream source publishes no LICENSE (see <code>models/README.md</code>),
        so they are never committed or deployed here. Without them there are
        no landmarks, pose, alignment or 1:1 compare.
      </p>
      <p>
        Generate the file locally, then drop it below (it stays in your
        browser — nothing is uploaded):
      </p>
      <pre class="weights-cmd"><code>git clone https://github.com/ruvnet/rvFACE
cd rvFACE &amp;&amp; python3 tools/fetch_and_convert.py</code></pre>
      <div class="weights-slots" data-role="slots"></div>
    `;

    const slotsHost = this.el.querySelector<HTMLElement>('[data-role="slots"]')!;
    const row = document.createElement('label');
    row.className = 'weights-slot';
    row.innerHTML = `
      <span class="ws-title"><code>${WEIGHT_FILES.landmark}</code> <span class="ws-desc">68-pt landmarks</span></span>
      <span class="ws-state" data-role="state">drop or pick file</span>
      <input type="file" accept=".safetensors" hidden />
    `;
    const input = row.querySelector<HTMLInputElement>('input')!;
    const state = row.querySelector<HTMLElement>('[data-role="state"]')!;

    const accept = async (file: File): Promise<void> => {
      let landmark: Uint8Array;
      try {
        landmark = await readSafetensorsFile(file, WEIGHT_FILES.landmark);
      } catch (err) {
        row.classList.remove('ws-ok');
        state.textContent = err instanceof Error ? err.message : String(err);
        state.classList.add('ws-err');
        this.status.log(
          `${WEIGHT_FILES.landmark}: ${err instanceof Error ? err.message : err}`,
          'error',
        );
        return;
      }
      row.classList.add('ws-ok');
      state.classList.remove('ws-err');
      state.textContent = `loaded ${file.name} ✓`;
      this.status.log(`landmark weights loaded (${file.name})`);
      this.onReady({ landmark });
    };

    input.addEventListener('change', () => {
      const f = input.files?.[0];
      if (f) void accept(f);
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
      if (f) void accept(f);
    });

    slotsHost.appendChild(row);
  }

  /** Insert the panel just above the analyze/compare panes. */
  mount(): void {
    if (!this.el.isConnected) this.host.querySelector('.panes')!.before(this.el);
  }

  /** Remove the panel (once the landmark is in and the engine restarts). */
  unmount(): void {
    this.el.remove();
  }
}
