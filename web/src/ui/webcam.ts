/**
 * Webcam controller: owns a `getUserMedia` stream bound to a `<video>` and
 * drives a backpressured render loop. `onFrame` runs once per animation
 * frame, but the next frame is scheduled only after it resolves, so a slow
 * engine throttles the loop instead of piling up work (ADR-0005).
 */

export class Webcam {
  private stream: MediaStream | null = null;
  private rafId = 0;

  constructor(
    private readonly video: HTMLVideoElement,
    /** Called per frame with the live `<video>`; must handle its own errors. */
    private readonly onFrame: (video: HTMLVideoElement) => Promise<void>,
  ) {}

  get active(): boolean {
    return this.stream !== null;
  }

  /** Source dimensions of the running stream (0 until metadata arrives). */
  get width(): number {
    return this.video.videoWidth;
  }
  get height(): number {
    return this.video.videoHeight;
  }

  /** Open the camera and start the frame loop. Throws if `getUserMedia` fails. */
  async start(): Promise<void> {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.rafId = requestAnimationFrame(() => void this.tick());
  }

  /** Stop the loop and release the camera. Idempotent. */
  stop(): void {
    if (!this.stream) return;
    cancelAnimationFrame(this.rafId);
    for (const track of this.stream.getTracks()) track.stop();
    this.stream = null;
    this.video.srcObject = null;
  }

  private async tick(): Promise<void> {
    if (!this.stream) return;
    // Skip until the first decoded frame gives real dimensions.
    if (this.video.videoWidth > 0 && this.video.videoHeight > 0) {
      await this.onFrame(this.video);
    }
    if (!this.stream) return; // stopped while awaiting onFrame
    this.rafId = requestAnimationFrame(() => void this.tick());
  }
}
