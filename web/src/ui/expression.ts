/**
 * Heuristic facial-expression guess from the 68 iBUG landmarks the pipeline
 * already produces — there is no trained expression model (the identity
 * embedder is deliberately expression-invariant). This reads pure geometry:
 * mouth openness, mouth-corner lift/width, brow raise and eye openness, each
 * normalized by the inter-ocular distance so it's scale/robust to distance.
 *
 * It's an estimate, not a classifier — good for smile / surprise / neutral,
 * intentionally conservative elsewhere. Thresholds are tuned against the
 * pipeline's own landmark scale (see `web` verification notes).
 */

import type { FaceResult } from '../engine';

export interface ExpressionGuess {
  /** Friendly label, e.g. "Smiling", "Surprised", "Neutral". */
  label: string;
  /** Rough 0–1 confidence in the guess. */
  confidence: number;
  /** Normalized metrics behind the guess (handy for tuning/debug). */
  metrics: ExpressionMetrics;
}

export interface ExpressionMetrics {
  /** Mouth aspect ratio: inner-lip opening ÷ mouth width. */
  mar: number;
  /** Mouth width ÷ inter-ocular distance (grows with a smile). */
  widthRatio: number;
  /** Mouth-corner lift above the lip midline ÷ IOD (>0 = smile, <0 = frown). */
  smileLift: number;
  /** Brow-to-eye gap ÷ IOD (grows when brows raise). */
  browRaise: number;
  /** Mean eye aspect ratio (low = closing/closed, high = wide). */
  ear: number;
}

const px = (lm: Float32Array, i: number): [number, number] => [lm[2 * i]!, lm[2 * i + 1]!];
const dist = (a: [number, number], b: [number, number]): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1]);

function meanPoint(lm: Float32Array, from: number, to: number): [number, number] {
  let x = 0;
  let y = 0;
  for (let i = from; i <= to; i++) {
    x += lm[2 * i]!;
    y += lm[2 * i + 1]!;
  }
  const n = to - from + 1;
  return [x / n, y / n];
}

function eyeAspect(lm: Float32Array, o: number): number {
  // Eye points o..o+5: corners o,o+3; upper o+1,o+2; lower o+5,o+4.
  const v = (dist(px(lm, o + 1), px(lm, o + 5)) + dist(px(lm, o + 2), px(lm, o + 4))) / 2;
  const h = dist(px(lm, o), px(lm, o + 3));
  return h > 1e-3 ? v / h : 0;
}

/** Compute the normalized geometry metrics for one face's landmarks. */
export function expressionMetrics(landmarks: Float32Array): ExpressionMetrics {
  const rightEye = meanPoint(landmarks, 36, 41);
  const leftEye = meanPoint(landmarks, 42, 47);
  const iod = Math.max(dist(leftEye, rightEye), 1e-3);

  const lc = px(landmarks, 48); // left mouth corner
  const rc = px(landmarks, 54); // right mouth corner
  const topOuter = px(landmarks, 51);
  const botOuter = px(landmarks, 57);
  const topInner = px(landmarks, 62);
  const botInner = px(landmarks, 66);

  const mouthW = Math.max(dist(lc, rc), 1e-3);
  const mar = dist(topInner, botInner) / mouthW;
  const widthRatio = mouthW / iod;
  const cornerY = (lc[1] + rc[1]) / 2;
  const lipMidY = (topOuter[1] + botOuter[1]) / 2;
  const smileLift = (lipMidY - cornerY) / iod; // canvas y grows down → corners up ⇒ >0

  const brow = meanPoint(landmarks, 17, 26);
  const eyeY = (leftEye[1] + rightEye[1]) / 2;
  const browRaise = (eyeY - brow[1]) / iod;

  const ear = (eyeAspect(landmarks, 36) + eyeAspect(landmarks, 42)) / 2;

  return { mar, widthRatio, smileLift, browRaise, ear };
}

/**
 * Guess an expression from landmarks. A small decision tree over the metrics,
 * ordered most-distinctive first.
 */
export function estimateExpression(landmarks: FaceResult['landmarks']): ExpressionGuess {
  // Detect-only engine mode produces empty landmark arrays — no estimate.
  if (landmarks.length < 136) {
    return {
      label: 'Unknown',
      confidence: 0,
      metrics: { mar: 0, widthRatio: 0, smileLift: 0, browRaise: 0, ear: 0 },
    };
  }
  const m = expressionMetrics(landmarks);
  const clamp = (v: number): number => Math.max(0.5, Math.min(0.95, v));

  // Thresholds calibrated against this pipeline's landmark scale: a real
  // neutral face measures mar≈0.09, smileLift≈0, browRaise≈0.27, ear≈0.35;
  // an open mouth reaches mar≈0.46 and a brow raise ≈0.45.

  // Eyes essentially shut.
  if (m.ear < 0.15) {
    return { label: 'Eyes closed', confidence: clamp(0.6 + (0.15 - m.ear) * 2), metrics: m };
  }
  // Open mouth: surprise if the brows also lift, otherwise just an open mouth.
  if (m.mar > 0.35) {
    if (m.browRaise > 0.4) {
      return { label: 'Surprised', confidence: clamp(0.55 + (m.mar - 0.35) + (m.browRaise - 0.4)), metrics: m };
    }
    return { label: 'Mouth open', confidence: clamp(0.55 + (m.mar - 0.35)), metrics: m };
  }
  // Smile (mouth not gaping): corners lifted, or the mouth clearly widened.
  if (m.smileLift > 0.045 || (m.widthRatio > 1.15 && m.smileLift > 0)) {
    const strength = Math.max((m.smileLift - 0.045) * 6, (m.widthRatio - 1.15) * 2);
    return { label: 'Smiling', confidence: clamp(0.6 + strength), metrics: m };
  }
  // Corners pulled down.
  if (m.smileLift < -0.03) {
    return { label: 'Frowning', confidence: clamp(0.55 + (-m.smileLift) * 4), metrics: m };
  }
  return { label: 'Neutral', confidence: 0.55, metrics: m };
}
