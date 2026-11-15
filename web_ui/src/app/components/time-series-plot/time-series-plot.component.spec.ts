import { ComponentFixture, TestBed } from '@angular/core/testing';
import { TimeSeriesPlotComponent } from './time-series-plot.component';
import {
  confidenceColor,
  crossesWrap,
  decimate,
  formatTime,
  niceTicks,
  frameAtX,
  segmentBetween,
  valueRange,
  zoomWindow,
} from './time-series-math';

describe('time-series math', () => {
  it('keeps a one-frame spike when many frames share a pixel', () => {
    const values = new Array(1000).fill(1);
    values[500] = 99;
    const { max } = decimate(values, { start: 0, end: 1000 }, 10);
    expect(max[5]).toBe(99);
    expect(max[4]).toBe(1);
  });

  it('marks columns without finite values as NaN', () => {
    const { min } = decimate([null, NaN, 3, 4], { start: 0, end: 4 }, 2);
    expect(Number.isNaN(min[0])).toBeTrue();
    expect(min[1]).toBe(3);
  });

  it('ignores missing values in the y range', () => {
    const [lo, hi] = valueRange(
      [{ label: 'a', values: [null, 2, 4, NaN] }],
      { start: 0, end: 4 },
    );
    expect(lo).toBeLessThan(2);
    expect(hi).toBeGreaterThan(4);
  });

  it('maps a pixel to the frame under it, within the window', () => {
    expect(frameAtX(0, 100, { start: 200, end: 300 })).toBe(200);
    expect(frameAtX(50, 100, { start: 200, end: 300 })).toBe(250);
    expect(frameAtX(100, 100, { start: 200, end: 300 })).toBe(299);
  });

  it('zooms around the anchor and stays inside the video', () => {
    const w = zoomWindow({ start: 0, end: 1000 }, 900, 0.5, 1000);
    expect(w.end - w.start).toBe(500);
    expect(w.end).toBeLessThanOrEqual(1000);
    expect(w.start).toBeLessThanOrEqual(900);
    expect(w.end).toBeGreaterThan(900);
  });

  it('colours low confidence red and high confidence near-white', () => {
    const rgb = (t: number) => confidenceColor(t).match(/\d+/g)!.map(Number);
    const [r0, g0, b0] = rgb(0.1);
    expect(r0).toBeGreaterThan(150);
    expect(g0).toBeLessThan(60);
    expect(b0).toBeLessThan(60);
    expect(Math.min(...rgb(0.95))).toBeGreaterThan(235);
  });

  it('tells a ±180° seam crossing from a real heading jump', () => {
    expect(crossesWrap(178, -179)).toBeTrue(); // a 3° turn
    expect(crossesWrap(-170, 170)).toBeTrue(); // a 20° turn
    expect(crossesWrap(130, -110)).toBeFalse(); // a 120° jump: drawn, not hidden
    expect(crossesWrap(10, 50)).toBeFalse();
  });

  it('puts ticks on round values', () => {
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(-155.5, 183.2, 4)).toEqual([-100, 0, 100]);
  });

  it('labels time like ethograph', () => {
    expect(formatTime(125, 60)).toBe('2:05');
    expect(formatTime(12, 5)).toBe('12s');
    expect(formatTime(1.5, 0.5)).toBe('1.5s');
  });

  it('builds an end-exclusive segment from two frames in any order', () => {
    expect(segmentBetween(30, 10)).toEqual([10, 31]);
  });
});

describe('TimeSeriesPlotComponent', () => {
  let fixture: ComponentFixture<TimeSeriesPlotComponent>;
  let component: TimeSeriesPlotComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TimeSeriesPlotComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(TimeSeriesPlotComponent);
    component = fixture.componentInstance;
    fixture.nativeElement.style.width = '474px'; // 64 px axis + 400 px plot + 10 px pad
    fixture.componentRef.setInput('series', [
      { label: 'nose', values: Array.from({ length: 400 }, (_, i) => i) },
    ]);
    fixture.componentRef.setInput('nFrames', 400);
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
    await fixture.whenStable();
  });

  afterEach(() => fixture.nativeElement.remove());

  function overlay(): HTMLCanvasElement {
    return fixture.nativeElement.querySelectorAll('canvas')[1];
  }

  function pointer(type: string, plotX: number) {
    const rect = overlay().getBoundingClientRect();
    overlay().dispatchEvent(
      new PointerEvent(type, {
        clientX: rect.left + 64 + plotX,
        button: 0,
        bubbles: true,
        pointerId: 1,
      }),
    );
  }

  it('emits the clicked frame', () => {
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    pointer('pointerdown', 200);
    pointer('pointerup', 200);
    expect(seen).toEqual([200]);
  });

  it('seeks the clicked frame in the heatmap, left of its colour bar', () => {
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    fixture.componentRef.setInput('mode', 'heatmap');
    fixture.detectChanges();
    // 474 px: 96 px row labels, 300 px plot, 78 px colour bar.
    const rect = overlay().getBoundingClientRect();
    const at = (type: string) =>
      overlay().dispatchEvent(
        new PointerEvent(type, { clientX: rect.left + 96 + 150, button: 0, bubbles: true, pointerId: 1 }),
      );
    at('pointerdown');
    at('pointerup');
    expect(seen).toEqual([200]);
  });

  it('pans time on a left drag, without seeking or marking', () => {
    component.xWindow.set({ start: 100, end: 300 });
    fixture.detectChanges();
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    pointer('pointerdown', 200);
    pointer('pointermove', 150);
    pointer('pointerup', 150);
    expect(seen).toEqual([]);
    expect(component.segment()).toBeNull();
    const w = component.xWindow()!;
    expect(w.start).toBeGreaterThan(100);
    expect(w.end - w.start).toBe(200);
  });

  it('marks a segment on drag instead of seeking when markSegment', () => {
    fixture.componentRef.setInput('markSegment', true);
    fixture.detectChanges();
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    pointer('pointerdown', 100);
    pointer('pointermove', 150);
    pointer('pointermove', 300);
    pointer('pointerup', 300);
    expect(seen).toEqual([]);
    expect(component.segment()).toEqual([100, 301]);
  });

  /** The visible frame window and value range, read from the component. */
  function view() {
    const c = component as unknown as {
      window: () => { start: number; end: number };
      yRange: () => [number, number];
    };
    const [lo, hi] = c.yRange();
    return { frames: c.window().end - c.window().start, values: hi - lo };
  }

  it('zooms time and values separately with a right-drag', () => {
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    const before = view();
    const rect = overlay().getBoundingClientRect();
    const at = (type: string, dx: number, dy: number) =>
      overlay().dispatchEvent(
        new PointerEvent(type, {
          clientX: rect.left + 264 + dx,
          clientY: rect.top + 60 + dy,
          button: 2,
          bubbles: true,
          pointerId: 1,
        }),
      );
    // Right only: time zooms in, values stay.
    at('pointerdown', 0, 0);
    at('pointermove', 100, 0);
    at('pointerup', 100, 0);
    const timeOnly = view();
    expect(timeOnly.frames).toBeLessThan(before.frames);
    expect(timeOnly.values).toBeCloseTo(before.values);
    // Up only: values zoom in, time stays.
    at('pointerdown', 0, 0);
    at('pointermove', 0, -100);
    at('pointerup', 0, -100);
    expect(view().frames).toBe(timeOnly.frames);
    expect(view().values).toBeLessThan(before.values);
    expect(seen).toEqual([]);
    expect(component.segment()).toBeNull();
  });

  it('keeps the current frame centred in follow mode, and Shift+wheel scrubs', () => {
    const seen: number[] = [];
    component.seek.subscribe((f) => seen.push(f));
    const rect = overlay().getBoundingClientRect();
    const wheel = (init: WheelEventInit) =>
      overlay().dispatchEvent(
        new WheelEvent('wheel', { clientX: rect.left + 264, clientY: rect.top + 50, bubbles: true, ...init }),
      );
    wheel({ deltaY: -100 }); // zoom in: 400 → 320 frames
    fixture.componentRef.setInput('follow', true);
    fixture.componentRef.setInput('currentFrame', 300);
    fixture.detectChanges();
    const view = () =>
      (component as unknown as { view: () => { start: number; end: number } }).view();
    expect(view()).toEqual({ start: 140, end: 460 });
    fixture.componentRef.setInput('currentFrame', 50);
    fixture.detectChanges();
    expect(view()).toEqual({ start: -110, end: 210 });
    wheel({ deltaY: 100, shiftKey: true });
    expect(seen).toEqual([82]); // 50 + 320 / 10
  });

  it('zooms only values when the wheel is over the value axis', () => {
    const before = view();
    const rect = overlay().getBoundingClientRect();
    overlay().dispatchEvent(
      new WheelEvent('wheel', {
        deltaY: -100,
        clientX: rect.left + 30,
        clientY: rect.top + 60,
        bubbles: true,
      }),
    );
    expect(view().frames).toBe(before.frames);
    expect(view().values).toBeLessThan(before.values);
  });

  it('holds a y range set from outside, and A goes back to autoscale', async () => {
    fixture.componentRef.setInput('yLimits', [0, 10]);
    fixture.detectChanges();
    await fixture.whenStable();
    expect(component.yRange()).toEqual([0, 10]);
    const reset = fixture.nativeElement.querySelector('button[aria-label="Show whole video"]');
    reset.click();
    expect(component.yLimits()).toBeNull();
    expect(component.yRange()).not.toEqual([0, 10]);
  });

  it('zooms the value axis with Ctrl+wheel and autoscales back', async () => {
    const reset = () =>
      fixture.nativeElement.querySelector('button[aria-label="Show whole video"]');
    const rect = overlay().getBoundingClientRect();
    overlay().dispatchEvent(
      new WheelEvent('wheel', {
        deltaY: -100,
        ctrlKey: true,
        clientX: rect.left + 264,
        clientY: rect.top + 50,
        bubbles: true,
      }),
    );
    fixture.detectChanges();
    await fixture.whenStable();
    reset().click();
    fixture.detectChanges();
    await fixture.whenStable();
    expect(reset()).toBeNull();
  });

  it('zooms with the wheel and offers to reset', async () => {
    const reset = () =>
      fixture.nativeElement.querySelector('button[aria-label="Show whole video"]');
    expect(reset()).toBeNull();
    const rect = overlay().getBoundingClientRect();
    overlay().dispatchEvent(
      new WheelEvent('wheel', { deltaY: -100, clientX: rect.left + 264, bubbles: true }),
    );
    fixture.detectChanges();
    await fixture.whenStable();
    expect(reset()).not.toBeNull();
    reset().click();
    fixture.detectChanges();
    await fixture.whenStable();
    expect(reset()).toBeNull();
  });
});
