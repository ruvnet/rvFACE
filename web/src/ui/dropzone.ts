/**
 * Drag-drop + click-to-pick image input. Turns `el` into an accessible
 * button: keyboard-focusable, activated by click / Enter / Space, and also a
 * drop target. Set `aria-disabled="true"` on `el` to suspend it (e.g. while a
 * live webcam owns the same view) — clicks, keys and drops are then ignored.
 */

export interface DropZoneOptions {
  /** Accessible name for the picker (also applied to the hidden file input). */
  ariaLabel: string;
}

export interface DropZoneHandle {
  /** Open the file picker programmatically (e.g. from an Upload button). */
  open(): void;
}

export function makeDropZone(
  el: HTMLElement,
  onFile: (file: File) => void,
  opts: DropZoneOptions,
): DropZoneHandle {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.hidden = true;
  input.setAttribute('aria-label', opts.ariaLabel);
  el.appendChild(input);

  el.setAttribute('role', 'button');
  el.setAttribute('tabindex', '0');
  el.setAttribute('aria-label', opts.ariaLabel);

  const disabled = () => el.getAttribute('aria-disabled') === 'true';

  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) onFile(file);
    input.value = '';
  });

  el.addEventListener('click', (e) => {
    if (disabled()) return;
    // Don't hijack clicks on interactive children (e.g. controls in the view).
    if ((e.target as HTMLElement).closest('button, input, a, video, label')) return;
    input.click();
  });

  el.addEventListener('keydown', (e) => {
    if (disabled()) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });

  el.addEventListener('dragover', (e) => {
    if (disabled()) return;
    e.preventDefault();
    el.classList.add('drag-over');
  });
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', (e) => {
    if (disabled()) return;
    e.preventDefault();
    el.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) onFile(file);
  });

  // An explicit Upload button opens the picker even when the zone itself is
  // suspended (aria-disabled) because a live webcam owns the view.
  return { open: () => input.click() };
}
