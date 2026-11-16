import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PredictionTracesComponent } from './prediction-traces.component';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { ListPoseSignalsResponse, TraceSource } from './pose-signals';
import { PredictionTracesSettingsService } from './prediction-traces-settings.service';
import { TraceRowComponent } from './trace-row.component';
import { By } from '@angular/platform-browser';
import { TimeSeriesPlotComponent } from '../../components/time-series-plot/time-series-plot.component';

const N = 100;

function source(): TraceSource {
  const data: Record<string, Float64Array> = {
    'nose|x': Float64Array.from({ length: N }, (_, i) => i),
    'nose|y': new Float64Array(N),
    'nose|likelihood': new Float64Array(N).fill(0.9),
    'ear_l|x': new Float64Array(N),
    'ear_r|x': new Float64Array(N),
  };
  return {
    key: 'm1/cam',
    modelKey: 'm1',
    viewName: 'cam',
    path: 'm1/video_preds/s.csv',
    keypoints: ['nose', 'ear_l', 'ear_r'],
    column: (kp, coord) => data[`${kp}|${coord}`] ?? null,
  };
}

const SIGNALS: ListPoseSignalsResponse = {
  movementAvailable: true,
  movementVersion: '0.17.0',
  signals: [
    {
      id: 'speed',
      label: 'Speed',
      function: 'movement.kinematics.compute_speed',
      docUrl: '',
      unit: 'px/s',
      params: [],
    },
    {
      id: 'head_direction',
      label: 'Head direction',
      function: 'movement.kinematics.compute_forward_vector_angle',
      docUrl: '',
      unit: 'deg',
      params: [
        { name: 'left_keypoint', label: 'Left keypoint', kind: 'keypoint' },
        { name: 'right_keypoint', label: 'Right keypoint', kind: 'keypoint' },
      ],
    },
  ],
};

describe('PredictionTracesComponent', () => {
  let fixture: ComponentFixture<PredictionTracesComponent>;
  let component: PredictionTracesComponent;
  let rpc: jasmine.SpyObj<RpcService>;

  beforeEach(async () => {
    localStorage.removeItem('prediction-traces.settings');
    rpc = jasmine.createSpyObj('RpcService', ['call']);
    rpc.call.and.callFake(async (method: string) => {
      if (method === 'listPoseSignals') return SIGNALS;
      if (method === 'computePoseSignal') {
        return {
          signal: 'speed',
          unit: 'px/s',
          nFrames: N,
          series: [{ label: 'nose', values: new Array(N).fill(1) }],
        };
      }
      throw new Error(method);
    });
    const projectInfo = jasmine.createSpyObj('ProjectInfoService', ['projectContext'], {
      projectInfo: { views: [] },
    });
    projectInfo.projectContext.and.returnValue({ key: 'demo' });

    await TestBed.configureTestingModule({
      imports: [PredictionTracesComponent],
      providers: [
        { provide: RpcService, useValue: rpc },
        { provide: ProjectInfoService, useValue: projectInfo },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(PredictionTracesComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('sources', [source()]);
    fixture.componentRef.setInput('nFrames', N);
    fixture.componentRef.setInput('fps', 30);
    fixture.componentRef.setInput('canExtract', true);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  });

  function select(label: string): HTMLSelectElement {
    return fixture.nativeElement.querySelector(`select[aria-label="${label}"]`);
  }

  async function choose(label: string, value: string) {
    const el = select(label);
    el.value = value;
    el.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  }

  function button(text: string): HTMLButtonElement | undefined {
    return Array.from(
      fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>,
    ).find((b) => b.textContent?.trim().startsWith(text));
  }

  function addButton(): HTMLButtonElement {
    return fixture.nativeElement.querySelector('button[aria-label="Add trace"]');
  }

  /** The i-th stacked trace row. */
  function row(i: number): TraceRowComponent {
    return fixture.debugElement.queryAll(By.directive(TraceRowComponent))[i]
      .componentInstance as TraceRowComponent;
  }

  async function settle() {
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  }

  it('lists movement signals next to the built-in ones', () => {
    const values = Array.from(select('Signal').options).map((o) => o.value);
    expect(values).toEqual(['position', 'confidence', 'speed', 'head_direction']);
  });

  it('computes a movement signal on the server for every keypoint', async () => {
    await choose('Signal', 'speed');
    expect(rpc.call).toHaveBeenCalledWith(
      'computePoseSignal',
      jasmine.objectContaining({
        projectKey: 'demo',
        predictionPath: 'm1/video_preds/s.csv',
        signal: 'speed',
        fps: 30,
        params: {},
      }),
    );
  });

  it('plots only the first keypoint at first', () => {
    const labels = () =>
      (row(0) as unknown as { series: () => { label: string }[] }).series().map((s) => s.label);
    expect(labels()).toEqual(['nose x', 'nose y']);
    const box = (kp: string) =>
      fixture.nativeElement.querySelector(`input[aria-label="Plot ${kp}"]`) as HTMLInputElement;
    expect(box('nose').checked).toBeTrue();
    expect(box('ear_l').checked).toBeFalse();
  });

  it('filters plotted keypoints and axes without recomputing', async () => {
    const labels = () =>
      (row(0) as unknown as { series: () => { label: string }[] }).series().map((s) => s.label);
    const box = (kp: string) =>
      fixture.nativeElement.querySelector(`input[aria-label="Plot ${kp}"]`) as HTMLInputElement;
    button('None')!.click();
    fixture.detectChanges();
    expect(labels()).toEqual([]);
    box('nose').click();
    fixture.detectChanges();
    expect(labels()).toEqual(['nose x', 'nose y']);
    await choose('Axes to plot', 'y');
    expect(labels()).toEqual(['nose']);
    box('ear_l').click();
    fixture.detectChanges();
    // ear_l has only an x column in this fixture, so y shows nose alone.
    expect(labels()).toEqual(['nose']);
    expect(rpc.call).not.toHaveBeenCalledWith('computePoseSignal', jasmine.anything());
  });

  it('keeps a typed y range per signal until auto is ticked', async () => {
    const plot = () =>
      fixture.debugElement.query(By.directive(TimeSeriesPlotComponent))
        .componentInstance as TimeSeriesPlotComponent;
    const field = (label: string) =>
      fixture.nativeElement.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
    const type = (label: string, value: string) => {
      field(label).value = value;
      field(label).dispatchEvent(new Event('change'));
      fixture.detectChanges();
    };
    expect(field('Autoscale y').checked).toBeTrue();
    type('Y minimum', '-5');
    type('Y maximum', '50');
    expect(plot().yRange()).toEqual([-5, 50]);
    expect(field('Autoscale y').checked).toBeFalse();
    // A max below the min is ignored.
    type('Y maximum', '-10');
    expect(plot().yRange()).toEqual([-5, 50]);
    // Holds while keypoints change and when coming back to the signal.
    button('None')!.click();
    fixture.detectChanges();
    expect(plot().yRange()).toEqual([-5, 50]);
    await choose('Signal', 'confidence');
    await choose('Signal', 'position');
    expect(plot().yRange()).toEqual([-5, 50]);
    await fixture.whenStable();
    expect(localStorage.getItem('prediction-traces.settings')).toContain('"position":[-5,50]');
    field('Autoscale y').click();
    fixture.detectChanges();
    expect(plot().yLimits()).toBeNull();
  });

  it('tells the Viewer which keypoints to draw the heading arrow from', async () => {
    const settings = TestBed.inject(PredictionTracesSettingsService);
    await choose('Signal', 'head_direction');
    expect(settings.headings()).toEqual([
      jasmine.objectContaining({ row: 0, left: 'ear_l', right: 'ear_r', cameraView: 'top_down' }),
    ]);
    await choose('Signal', 'position');
    expect(settings.headings()).toEqual([]);
  });

  it('keeps two head-direction rows independent, one arrow each', () => {
    const settings = TestBed.inject(PredictionTracesSettingsService);
    settings.updateTrace(0, { signalId: 'head_direction' });
    settings.addTrace();
    settings.updateTrace(1, { signalId: 'head_direction' });
    settings.setParams(0, 'head_direction', { left_keypoint: 'ear_l', right_keypoint: 'ear_r' });
    settings.setParams(1, 'head_direction', { left_keypoint: 'nose', right_keypoint: 'ear_r' });
    settings.setYLimits(0, 'head_direction', [-90, 90]);
    const [a, b] = settings.traces();
    expect(a.paramsBySignal['head_direction']['left_keypoint']).toBe('ear_l');
    expect(b.paramsBySignal['head_direction']['left_keypoint']).toBe('nose');
    expect(b.yLimitsBySignal['head_direction']).toBeUndefined();
    const headings = settings.headings();
    expect(headings.map((h) => h.left)).toEqual(['ear_l', 'nose']);
    expect(headings[0].color).not.toBe(headings[1].color);
  });

  it('keeps the choices when the panel is closed and opened again', async () => {
    // Position: plot ear_l instead of the default first keypoint.
    button('None')!.click();
    fixture.detectChanges();
    (fixture.nativeElement.querySelector('input[aria-label="Plot ear_l"]') as HTMLInputElement).click();
    fixture.detectChanges();
    await choose('Signal', 'head_direction');
    await choose('Right keypoint', 'nose');
    fixture.destroy();

    // Reopen: a new panel for the same app.
    fixture = TestBed.createComponent(PredictionTracesComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('sources', [source()]);
    fixture.componentRef.setInput('nFrames', N);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    expect(select('Signal').value).toBe('head_direction');
    expect(select('Right keypoint').value).toBe('nose');
    await choose('Signal', 'position');
    const box = (kp: string) =>
      fixture.nativeElement.querySelector(`input[aria-label="Plot ${kp}"]`) as HTMLInputElement;
    expect(box('ear_l').checked).toBeTrue();
    expect(box('nose').checked).toBeFalse();
  });

  it('centres on the frame by default; turning it off is remembered', () => {
    const toggle = fixture.nativeElement.querySelector(
      'input[aria-label="Centre on current frame"]',
    ) as HTMLInputElement;
    expect(toggle.checked).toBeTrue();
    toggle.click();
    fixture.detectChanges();
    const plot = fixture.debugElement.query(By.directive(TimeSeriesPlotComponent))
      .componentInstance as TimeSeriesPlotComponent;
    expect(plot.follow()).toBeFalse();
    expect(TestBed.inject(PredictionTracesSettingsService).followPlayback()).toBeFalse();
  });

  it('keeps the choices across a page reload', async () => {
    const settings = TestBed.inject(PredictionTracesSettingsService);
    settings.updateTrace(0, { signalId: 'confidence', spaceChoice: 'y' });
    settings.addTrace();
    settings.update({ selection: 'uniform' });
    await fixture.whenStable();
    const reloaded = TestBed.runInInjectionContext(() => new PredictionTracesSettingsService());
    expect(reloaded.traces().map((t) => t.signalId)).toEqual(['confidence', 'position']);
    expect(reloaded.traces()[0].spaceChoice).toBe('y');
    expect(reloaded.selection()).toBe('uniform');
  });

  it('reads settings saved before there were rows as row 1', () => {
    localStorage.setItem(
      'prediction-traces.settings',
      JSON.stringify({ signalId: 'speed', spaceChoice: 'x', selection: 'uniform' }),
    );
    const reloaded = TestBed.runInInjectionContext(() => new PredictionTracesSettingsService());
    expect(reloaded.traces().length).toBe(1);
    expect(reloaded.traces()[0]).toEqual(
      jasmine.objectContaining({ signalId: 'speed', spaceChoice: 'x' }),
    );
    expect(reloaded.selection()).toBe('uniform');
  });

  it('adds up to three trace rows, each with its own signal, and closes them', async () => {
    const signals = () =>
      Array.from(
        fixture.nativeElement.querySelectorAll('select[aria-label="Signal"]') as NodeListOf<HTMLSelectElement>,
      ).map((s) => s.value);
    expect(signals()).toEqual(['position']);
    expect(fixture.nativeElement.querySelector('button[aria-label="Close this trace"]')).toBeNull();
    addButton().click();
    await settle();
    expect(signals()).toEqual(['position', 'confidence']);
    addButton().click();
    await settle();
    expect(signals().length).toBe(3);
    expect(addButton().disabled).toBeTrue();
    (fixture.nativeElement.querySelectorAll('button[aria-label="Close this trace"]')[1] as HTMLButtonElement).click();
    await settle();
    expect(signals()).toEqual(['position', 'position']);
    expect(addButton().disabled).toBeFalse();
  });

  it('zooms every row together and shares the segment', async () => {
    addButton().click();
    await settle();
    const plots = () =>
      fixture.debugElement
        .queryAll(By.directive(TimeSeriesPlotComponent))
        .map((d) => d.componentInstance as TimeSeriesPlotComponent);
    plots()[0].xWindow.set({ start: 10, end: 50 });
    await settle();
    expect(plots()[1].xWindow()).toEqual({ start: 10, end: 50 });
    plots()[1].segment.set([20, 30]);
    await settle();
    expect(component.segment()).toEqual([20, 30]);
    expect(plots()[0].segment()).toEqual([20, 30]);
  });

  it('guesses left and right keypoints for head direction', async () => {
    await choose('Signal', 'head_direction');
    expect(select('Left keypoint').value).toBe('ear_l');
    expect(select('Right keypoint').value).toBe('ear_r');
    expect(rpc.call).toHaveBeenCalledWith(
      'computePoseSignal',
      jasmine.objectContaining({
        params: { left_keypoint: 'ear_l', right_keypoint: 'ear_r' },
      }),
    );
  });

  it('extracts the requested number of frames from the segment, diverse or uniform', async () => {
    const emitted: unknown[] = [];
    component.extractSegment.subscribe((e) => emitted.push(e));
    button('Extract from segment')!.click();
    component.segment.set([20, 60]);
    fixture.detectChanges();
    await fixture.whenStable();
    button('Extract frames')!.click();
    expect(emitted).toEqual([{ segment: [20, 60], nFrames: 10, selection: 'kmeans' }]);
    await choose('How to pick frames', 'uniform');
    button('Extract frames')!.click();
    expect(emitted.at(-1)).toEqual({ segment: [20, 60], nFrames: 10, selection: 'uniform' });
  });

  it('blocks extracting more frames than the segment holds', async () => {
    button('Extract from segment')!.click();
    component.segment.set([20, 25]);
    fixture.detectChanges();
    await fixture.whenStable();
    expect(button('Extract frames')!.disabled).toBeTrue();
  });

  it('sets the segment from the current frame', async () => {
    button('Extract from segment')!.click();
    fixture.detectChanges();
    fixture.componentRef.setInput('currentFrame', 30);
    fixture.detectChanges();
    button('Start at current frame')!.click();
    fixture.componentRef.setInput('currentFrame', 70);
    fixture.detectChanges();
    button('End at current frame')!.click();
    expect(component.segment()).toEqual([30, 71]);
  });

  it('left-drag pans until Extract from segment is clicked, and Done clears the segment', async () => {
    const plot = () =>
      fixture.debugElement.query(By.directive(TimeSeriesPlotComponent))
        .componentInstance as TimeSeriesPlotComponent;
    expect(plot().markSegment()).toBeFalse();
    expect(button('Start at current frame')).toBeUndefined();
    button('Extract from segment')!.click();
    await settle();
    expect(plot().markSegment()).toBeTrue();
    component.segment.set([20, 60]);
    await settle();
    button('Done')!.click();
    await settle();
    expect(plot().markSegment()).toBeFalse();
    expect(component.segment()).toBeNull();
    expect(button('Start at current frame')).toBeUndefined();
  });
});
