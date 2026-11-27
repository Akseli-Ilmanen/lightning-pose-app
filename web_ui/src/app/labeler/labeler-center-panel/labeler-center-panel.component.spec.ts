import { Component, input, model, output, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

import { LabelerCenterPanelComponent } from './labeler-center-panel.component';
import { ImageLabelWidgetComponent } from '../image-label-widget/image-label-widget.component';
import { ProjectInfoService } from '../../project-info.service';
import { CameraCalibrationService } from '../../camera-calibration.service';
import { MVFrame } from '../frame.model';
import { ContextSlotView } from '../temporal-context';
import { LabelerViewOptionsService } from '../labeler-view-options.service';
import { SessionService } from '../../session.service';
import { RpcService } from '../../rpc.service';
import { AuditFocus, ShapeGuides, ShapePair } from '../label-audit/label-audit.model';
import { LabelAuditService } from '../label-audit/label-audit.service';

describe('LabelerCenterPanelComponent temporal context', () => {
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;

  beforeEach(async () => {
    const projectInfo = jasmine.createSpyObj(
      'ProjectInfoService',
      ['projectContext'],
      { projectInfo: { data_dir: 'data', views: [] } },
    );
    projectInfo.projectContext.and.returnValue(null);
    const calibration = jasmine.createSpyObj('CameraCalibrationService', [
      'getCalibrationStatus',
    ]);
    calibration.getCalibrationStatus.and.resolveTo('none');

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibration },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: [{ viewName: 'main', csvPath: 'CollectedData.csv' }],
    });
  });

  function showFrame(imgPath: string) {
    const frame: MVFrame = {
      key: imgPath,
      views: [
        { viewName: 'main', imgPath, keypoints: [], originalKeypoints: [] },
      ],
    };
    fixture.componentRef.setInput('frame', frame);
    fixture.detectChanges();
  }

  function button(label: string): HTMLButtonElement {
    return fixture.nativeElement.querySelector(`button[aria-label="${label}"]`);
  }

  function contextLabel(): string {
    return fixture.nativeElement
      .querySelector('[data-testid="context-frame-label"]')
      .textContent.trim();
  }

  function click(label: string) {
    button(label).click();
    fixture.detectChanges();
  }

  afterEach(() => viewOptions().pauseTemporalContext());

  function viewOptions(): LabelerViewOptionsService {
    return fixture.debugElement.injector.get(LabelerViewOptionsService);
  }

  it('opens a frame paused on the current frame', () => {
    showFrame('labeled-data/s/img00000042.png');
    expect(button('Play temporal context').disabled).toBeFalse();
    expect(contextLabel()).toBe('Showing: current');
    expect(
      button('Show context frame current').getAttribute('aria-pressed'),
    ).toBe('true');
    expect(
      button(
        'Play temporal context',
      ).parentElement!.parentElement!.getAttribute('data-tip'),
    ).toBeNull();
  });

  it('steps one context frame at a time and stops at the ends', () => {
    showFrame('labeled-data/s/img00000042.png');
    click('Next context frame');
    expect(contextLabel()).toBe('Showing: +1');
    click('Next context frame');
    expect(contextLabel()).toBe('Showing: +2');
    expect(button('Next context frame').disabled).toBeTrue();
    click('Previous context frame');
    expect(contextLabel()).toBe('Showing: +1');
  });

  it('shows a Back to the current frame button only away from the current frame', () => {
    showFrame('labeled-data/s/img00000042.png');
    const reset = () => button('Back to the current frame');
    expect(reset()).toBeNull();

    click('Show context frame +2');
    expect(reset().classList).toContain('btn-warning');
    expect(reset().textContent!.trim()).toBe('restart_alt');
    const tip = reset().closest('[data-tip]')!;
    expect(tip.getAttribute('data-tip')).toBe('Back to the current frame');
    // Opens to the left: centered, the right side panel's edge cuts it off.
    expect(tip.classList).toContain('tooltip-left');
    click('Back to the current frame');
    expect(contextLabel()).toBe('Showing: current');
    expect(reset()).toBeNull();
  });

  it('shows the Back button while playing and stops playback with one click', () => {
    jasmine.clock().install();
    try {
      showFrame('labeled-data/s/img00000042.png');
      click('Play temporal context');
      jasmine.clock().tick(1000);
      fixture.detectChanges();
      click('Back to the current frame');
      expect(viewOptions().isPlayingTemporalContext()).toBeFalse();
      expect(contextLabel()).toBe('Showing: current');
      expect(button('Back to the current frame')).toBeNull();
    } finally {
      jasmine.clock().uninstall();
    }
  });

  it('jumps to a context frame when its dot is clicked', () => {
    showFrame('labeled-data/s/img00000042.png');
    click('Show context frame −2');
    expect(contextLabel()).toBe('Showing: −2');
    expect(button('Previous context frame').disabled).toBeTrue();
    expect(button('Show context frame −2').getAttribute('aria-pressed')).toBe(
      'true',
    );
  });

  it('plays, and Pause holds the visible frame', () => {
    jasmine.clock().install();
    try {
      showFrame('labeled-data/s/img00000042.png');
      click('Play temporal context');
      expect(button('Pause temporal context')).not.toBeNull();
      jasmine.clock().tick(300);
      fixture.detectChanges();
      expect(contextLabel()).toBe('Showing: +1');
      click('Pause temporal context');
      jasmine.clock().tick(900);
      fixture.detectChanges();
      expect(contextLabel()).toBe('Showing: +1');
      expect(button('Play temporal context')).not.toBeNull();
    } finally {
      jasmine.clock().uninstall();
    }
  });

  it('returns to the current frame when another frame opens', () => {
    showFrame('labeled-data/s/img00000042.png');
    click('Show context frame +2');
    showFrame('labeled-data/s/img00000043.png');
    expect(contextLabel()).toBe('Showing: current');
  });

  function widget(): ImageLabelWidgetStubComponent {
    return fixture.debugElement.query(
      By.directive(ImageLabelWidgetStubComponent),
    ).componentInstance;
  }

  function expectControlsDisabled() {
    for (const label of [
      'Play temporal context',
      'Previous context frame',
      'Next context frame',
      'Show context frame +1',
    ]) {
      expect(button(label).disabled).withContext(label).toBeTrue();
    }
  }

  function tip(): string | null {
    return button(
      'Play temporal context',
    ).parentElement!.parentElement!.getAttribute('data-tip');
  }

  it('disables the controls and says so when the context frames are not found', () => {
    showFrame('labeled-data/s/img00000042.png');
    expect(button('Play temporal context').disabled).toBeFalse();
    widget().contextFramesMissing.set(true);
    fixture.detectChanges();
    expectControlsDisabled();
    expect(tip()).toBe('Context frames were not found.');
  });

  it('stops playback and returns to the current frame when the context frames are not found', () => {
    showFrame('labeled-data/s/img00000042.png');
    click('Play temporal context');
    widget().contextFramesMissing.set(true);
    fixture.detectChanges();
    expect(viewOptions().isPlayingTemporalContext()).toBeFalse();
    expect(contextLabel()).toBe('Showing: current');
  });

  it('says the current frame is shown when a context frame was not found', () => {
    showFrame('labeled-data/s/img00000099.png');
    widget().contextSlotViews.set([
      { shownOffset: -2, substitute: null },
      { shownOffset: -1, substitute: null },
      { shownOffset: 0, substitute: null },
      { shownOffset: 1, substitute: null },
      { shownOffset: 0, substitute: 'not-found' },
    ]);
    click('Show context frame +1');
    expect(contextLabel()).toBe('Showing: +1');
    click('Show context frame +2');
    expect(contextLabel()).toBe('Showing: current in place of +2 (not found)');
  });

  it('says which frame is shown before the start of the video', () => {
    showFrame('labeled-data/s/img00000000.png');
    widget().contextSlotViews.set([
      { shownOffset: 0, substitute: 'before-start' },
      { shownOffset: 0, substitute: 'before-start' },
      { shownOffset: 0, substitute: null },
      { shownOffset: 1, substitute: null },
      { shownOffset: 2, substitute: null },
    ]);
    click('Show context frame −1');
    expect(contextLabel()).toBe(
      'Showing: current in place of −1 (before the video starts)',
    );
  });

  it("says the frame's own image could not be loaded before blaming context frames", () => {
    showFrame('labeled-data/s/img00000042.png');
    widget().currentFrameMessage.set("This frame's image could not be loaded.");
    widget().contextFramesMissing.set(true);
    fixture.detectChanges();
    expectControlsDisabled();
    expect(tip()).toBe("This frame's image could not be loaded.");
  });

  it('says why a TIFF frame has no temporal context', () => {
    showFrame('labeled-data/s/img00000042.tif');
    widget().currentFrameMessage.set(
      'TIFF frames train in Lightning Pose but cannot be shown in the browser. Convert them to PNG to label or review them here.',
    );
    fixture.detectChanges();
    expectControlsDisabled();
    expect(tip()).toContain('TIFF frames train in Lightning Pose');
  });

  it('disables the controls and explains why when the file name has no frame number', () => {
    showFrame('labeled-data/s/frame_a.png');
    expectControlsDisabled();
    expect(tip()).toBe('The image file name has no frame number.');
  });
});

describe('LabelerCenterPanelComponent side panel', () => {
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;

  beforeEach(async () => {
    const projectInfo = jasmine.createSpyObj(
      'ProjectInfoService',
      ['projectContext'],
      { projectInfo: { data_dir: 'data', views: [] } },
    );
    projectInfo.projectContext.and.returnValue(null);
    const calibration = jasmine.createSpyObj('CameraCalibrationService', [
      'getCalibrationStatus',
    ]);
    calibration.getCalibrationStatus.and.resolveTo('none');

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibration },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: [{ viewName: 'main', csvPath: 'CollectedData.csv' }],
    });
    const frame: MVFrame = {
      key: 'f',
      views: [
        {
          viewName: 'main',
          imgPath: 'labeled-data/s/img00000042.png',
          keypoints: [],
          originalKeypoints: [],
        },
      ],
    };
    fixture.componentRef.setInput('frame', frame);
    fixture.detectChanges();
  });

  function slider(label: string): HTMLInputElement {
    return fixture.nativeElement.querySelector(
      `input[type="range"][aria-label="${label}"]`,
    );
  }

  it('puts Save, Next and Revert right under the keypoint list, before View options', () => {
    const el: HTMLElement = fixture.nativeElement;
    const titles = Array.from(el.querySelectorAll('.app-accordion-title h1'));
    const keypoints = titles.find((h) => h.textContent!.trim() === 'Keypoints')!;
    const viewOptions = titles.find(
      (h) => h.textContent!.trim() === 'View options',
    )!;
    const actions = el.querySelector('[data-testid="frame-actions"]')!;
    const before = Node.DOCUMENT_POSITION_PRECEDING;
    const after = Node.DOCUMENT_POSITION_FOLLOWING;
    expect(actions.compareDocumentPosition(keypoints) & before).toBeTruthy();
    expect(actions.compareDocumentPosition(viewOptions) & after).toBeTruthy();
    expect(actions.textContent).toContain('Save and next');
    expect(actions.textContent).toContain('Next →');
    expect(actions.textContent).toContain('Revert changes');
  });

  it('puts Keep selected and Clear values above the keypoint list, outside the header', () => {
    const el: HTMLElement = fixture.nativeElement;
    const tools = el.querySelector('[data-testid="keypoint-tools"]')!;
    const header = el.querySelector('label[for="app-accordion-keypoints"]')!;
    const list = el.querySelector('.panel-content ul')!;
    expect(
      tools.querySelector('input[aria-label="Keep selected keypoint"]'),
    ).not.toBeNull();
    expect(
      tools.querySelector('button[aria-label="Clear values in this view"]'),
    ).not.toBeNull();
    expect(header.querySelector('button')).toBeNull();
    expect(
      tools.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      el.querySelectorAll('input[aria-label="Keep selected keypoint"]').length,
    ).toBe(1);
  });

  it('groups View options into Context, Image, Keypoints and Skeleton, with the pixel grid on its own', () => {
    const el: HTMLElement = fixture.nativeElement;
    const groups = Array.from(el.querySelectorAll('[role="group"]'));
    expect(groups.map((g) => g.getAttribute('aria-label'))).toEqual([
      'Context',
      'Image',
      'Keypoints',
      'Skeleton',
    ]);
    const group = (name: string) =>
      groups.find((g) => g.getAttribute('aria-label') === name)!;
    expect(
      group('Context').querySelector('[aria-label="Play temporal context"]'),
    ).not.toBeNull();
    for (const label of ['Image size', 'Image brightness', 'Image contrast']) {
      expect(group('Image').querySelector(`[aria-label="${label}"]`))
        .withContext(label)
        .not.toBeNull();
    }
    for (const label of [
      'Keypoint opacity',
      'Keypoint size',
      'Keypoint labels',
    ]) {
      expect(group('Keypoints').querySelector(`[aria-label="${label}"]`))
        .withContext(label)
        .not.toBeNull();
    }
    expect(
      group('Skeleton').querySelector('[aria-label="Show skeleton"]'),
    ).not.toBeNull();
    const grid = el.querySelector('input[aria-label="Pixel grid overlay"]')!;
    expect(grid.closest('[role="group"]')).toBeNull();
    expect(
      group('Skeleton').compareDocumentPosition(grid) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('shows Show skeleton off when the project defines no skeleton pairs', () => {
    // This describe's project info has no skeleton key.
    const toggle: HTMLInputElement = fixture.nativeElement.querySelector(
      'input[aria-label="Show skeleton"]',
    );
    expect(toggle.checked).toBeFalse();
    expect(toggle.disabled).toBeTrue();
  });

  it('Show slider numbers shows the View options values only while on', () => {
    const options = fixture.debugElement.injector.get(
      LabelerViewOptionsService,
    );
    const before = options.showSliderNumbers();
    const KEY = 'labeler-view-options.showSliderNumbers';
    const stored = localStorage.getItem(KEY);
    const el: HTMLElement = fixture.nativeElement;
    const imageGroup = () => el.querySelector('[role="group"][aria-label="Image"]')!;
    const toggle: HTMLInputElement = el.querySelector(
      'input[aria-label="Show slider numbers"]',
    )!;
    try {
      options.showSliderNumbers.set(false);
      fixture.detectChanges();
      expect(toggle.checked).toBeFalse();
      expect(imageGroup().textContent).not.toContain(
        `${options.imageSizePx()}px`,
      );

      toggle.click();
      fixture.detectChanges();
      expect(options.showSliderNumbers()).toBeTrue();
      expect(imageGroup().textContent).toContain(`${options.imageSizePx()}px`);
    } finally {
      options.showSliderNumbers.set(before);
      fixture.detectChanges();
      if (stored === null) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, stored);
    }
  });

  it('shows every switch in the side panel in the primary (green) style', () => {
    const el: HTMLElement = fixture.nativeElement;
    const toggles: HTMLInputElement[] = Array.from(
      el.querySelectorAll('input.toggle'),
    );
    expect(toggles.map((t) => t.getAttribute('aria-label'))).toEqual([
      'Keep selected keypoint',
      'Keypoint labels',
      'Show skeleton',
      'Pixel grid overlay',
      'Show slider numbers',
    ]);
    for (const t of toggles) {
      expect(t.classList)
        .withContext(t.getAttribute('aria-label')!)
        .toContain('toggle-primary');
    }
  });

  it('lets the image and keypoint sliders reach their wider limits', () => {
    expect([slider('Image size').min, slider('Image size').max]).toEqual([
      '200',
      '1600',
    ]);
    expect([
      slider('Image brightness').min,
      slider('Image brightness').max,
    ]).toEqual(['0', '2']);
    expect([slider('Image contrast').min, slider('Image contrast').max]).toEqual(
      ['0', '2'],
    );
    expect(slider('Keypoint opacity').max).toBe('1');
  });
});

describe('LabelerCenterPanelComponent Next and keypoint selection', () => {
  const KEEP_KEY = 'labeler-view-options.keepSelectedKeypoint';
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;
  let session: jasmine.SpyObj<SessionService>;
  let saved: string | null;

  const NAMES = ['nose', 'ear', 'tail'];

  /** A frame with the given views; image files are named <key>_<view>.png. */
  function makeFrame(
    key: string,
    views: string[],
    opts: { unlabeled?: boolean; editedView?: string } = {},
  ): MVFrame {
    return {
      key,
      views: views.map((viewName) => {
        const keypoints = NAMES.map((keypointName, i) => ({
          keypointName,
          x: opts.unlabeled ? NaN : i,
          y: opts.unlabeled ? NaN : i,
        }));
        return {
          viewName,
          imgPath: `labeled-data/s/${key}_${viewName}.png`,
          keypoints:
            viewName === opts.editedView
              ? keypoints.map((kp) => ({ ...kp, x: kp.x + 5 }))
              : keypoints,
          originalKeypoints: opts.unlabeled ? null : keypoints,
        };
      }),
    };
  }

  beforeEach(async () => {
    saved = localStorage.getItem(KEEP_KEY);
    localStorage.removeItem(KEEP_KEY);
    const projectInfo = jasmine.createSpyObj(
      'ProjectInfoService',
      ['projectContext'],
      { projectInfo: { data_dir: 'data', views: [] } },
    );
    projectInfo.projectContext.and.returnValue(null);
    const calibration = jasmine.createSpyObj('CameraCalibrationService', [
      'getCalibrationStatus',
    ]);
    calibration.getCalibrationStatus.and.resolveTo('none');
    session = jasmine.createSpyObj('SessionService', ['saveMVFrame']);
    session.saveMVFrame.and.resolveTo();

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibration },
        { provide: SessionService, useValue: session },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: [{ viewName: 'top', csvPath: 'CollectedData_top.csv' }],
    });
  });

  afterEach(() => {
    if (saved === null) localStorage.removeItem(KEEP_KEY);
    else localStorage.setItem(KEEP_KEY, saved);
  });

  function open(frame: MVFrame, hasNextFrame = true) {
    fixture.componentRef.setInput('hasNextFrame', hasNextFrame);
    fixture.componentRef.setInput('frame', frame);
    fixture.detectChanges();
  }

  function actionButton(text: string): HTMLButtonElement {
    const buttons: HTMLButtonElement[] = Array.from(
      fixture.nativeElement.querySelectorAll(
        '[data-testid="frame-actions"] button',
      ),
    );
    return buttons.find((b) => b.textContent!.includes(text))!;
  }

  function clickNext() {
    actionButton('Next →').click();
    fixture.detectChanges();
  }

  /** File name of the image in the main view's header, e.g. "f1_side.png". */
  function shownImage(): string {
    const spans: HTMLElement[] = Array.from(
      fixture.nativeElement.querySelectorAll('span.truncate'),
    );
    return spans
      .map((el) => el.textContent!.trim())
      .find((text) => !text.includes('/'))!;
  }

  function keypointRow(name: string): HTMLElement {
    const rows: HTMLElement[] = Array.from(
      fixture.nativeElement.querySelectorAll('li'),
    );
    return rows.find(
      (li) => li.querySelector('span')!.textContent!.trim() === name,
    )!;
  }

  /** Name of the highlighted row in the keypoint list. */
  function selectedKeypoint(): string {
    return fixture.nativeElement
      .querySelector('li.bg-sky-700 span')
      .textContent.trim();
  }

  function selectKeypoint(name: string) {
    keypointRow(name).click();
    fixture.detectChanges();
  }

  it('Next walks the views of a frame, then opens the next frame on its first view, without saving', () => {
    const nextFrame = jasmine.createSpy('nextFrame');
    fixture.componentInstance.nextFrame.subscribe(nextFrame);
    open(makeFrame('f1', ['top', 'side', 'front']));
    expect(shownImage()).toBe('f1_top.png');

    clickNext();
    expect(shownImage()).toBe('f1_side.png');
    clickNext();
    expect(shownImage()).toBe('f1_front.png');
    expect(nextFrame).not.toHaveBeenCalled();

    clickNext();
    expect(nextFrame).toHaveBeenCalledTimes(1);
    open(makeFrame('f2', ['top', 'side', 'front']));
    expect(shownImage()).toBe('f2_top.png');
    expect(session.saveMVFrame).not.toHaveBeenCalled();
  });

  it('Next opens the next frame from a single view, and is inactive on the last view of the last frame', () => {
    const nextFrame = jasmine.createSpy('nextFrame');
    fixture.componentInstance.nextFrame.subscribe(nextFrame);
    open(makeFrame('f1', ['top']));
    clickNext();
    expect(nextFrame).toHaveBeenCalledTimes(1);

    open(makeFrame('f2', ['top', 'side']), false);
    expect(actionButton('Next →').classList).not.toContain('btn-disabled');
    clickNext();
    expect(shownImage()).toBe('f2_side.png');
    expect(actionButton('Next →').classList).toContain('btn-disabled');
  });

  it('frame buttons have a tooltip only while inactive, saying why; Delete always has one', () => {
    const tip = (text: string) =>
      actionButton(text).closest('.tooltip')?.getAttribute('data-tip') ?? null;
    const deleteTip = () =>
      fixture.nativeElement
        .querySelector('[data-testid="frame-actions"] .btn-error')
        .closest('.tooltip')
        .getAttribute('data-tip');

    // Edited, with frames before and after: everything is active.
    fixture.componentRef.setInput('hasPreviousFrame', true);
    open(makeFrame('f1', ['top'], { editedView: 'top' }));
    for (const text of ['Save', 'Save and next', '← Back', 'Next →', 'Revert changes']) {
      expect(tip(text)).withContext(text).toBeNull();
    }
    expect(deleteTip()).toBeTruthy();

    // Unedited, and the only frame: all but Delete are inactive.
    fixture.componentRef.setInput('hasPreviousFrame', false);
    open(makeFrame('f2', ['top']), false);
    for (const text of ['Save', 'Save and next', '← Back', 'Next →', 'Revert changes']) {
      expect(actionButton(text).classList).withContext(text).toContain('btn-disabled');
      expect(tip(text)).withContext(text).toBeTruthy();
    }
    expect(tip('Save')).toBe('No changes to save.');
    expect(tip('Next →')).toBe('Already on the last view of the last frame.');
    expect(tip('← Back')).toBe('Already on the first view of the first frame.');
    expect(deleteTip()).toBeTruthy();
  });

  it('Save and next from the last view opens the next frame on its first view', async () => {
    const events: { shouldAdvanceFrame: boolean }[] = [];
    fixture.componentInstance.saved.subscribe((e) => events.push(e));
    open(makeFrame('f1', ['top', 'side'], { editedView: 'side' }));
    clickNext();
    expect(shownImage()).toBe('f1_side.png');

    actionButton('Save and next').click();
    // Let the save promise and its .then() run.
    await session.saveMVFrame.calls.mostRecent().returnValue;
    await Promise.resolve();
    expect(session.saveMVFrame).toHaveBeenCalledTimes(1);
    expect(events[0].shouldAdvanceFrame).toBeTrue();

    open(makeFrame('f2', ['top', 'side']));
    expect(shownImage()).toBe('f2_top.png');
  });

  it('reports saving while a save is written, and done when it finishes', async () => {
    const saving: boolean[] = [];
    fixture.componentInstance.savingChange.subscribe((v) => saving.push(v));
    open(makeFrame('f1', ['top'], { editedView: 'top' }));

    actionButton('Save').click();
    expect(saving).toEqual([true]);
    await session.saveMVFrame.calls.mostRecent().returnValue;
    await Promise.resolve();
    expect(saving).toEqual([true, false]);
  });

  it('Back walks the views back, then opens the previous frame on its last view, without saving', () => {
    const previousFrame = jasmine.createSpy('previousFrame');
    fixture.componentInstance.previousFrame.subscribe(previousFrame);
    fixture.componentRef.setInput('hasPreviousFrame', true);
    open(makeFrame('f2', ['top', 'side', 'front']));
    clickNext();
    clickNext();
    expect(shownImage()).toBe('f2_front.png');

    actionButton('← Back').click();
    fixture.detectChanges();
    expect(shownImage()).toBe('f2_side.png');
    actionButton('← Back').click();
    fixture.detectChanges();
    expect(shownImage()).toBe('f2_top.png');
    expect(previousFrame).not.toHaveBeenCalled();

    actionButton('← Back').click();
    fixture.detectChanges();
    expect(previousFrame).toHaveBeenCalledTimes(1);
    open(makeFrame('f1', ['top', 'side', 'front']));
    expect(shownImage()).toBe('f1_front.png');
    expect(session.saveMVFrame).not.toHaveBeenCalled();

    // On the first view of the first frame there is nowhere to go back to.
    fixture.componentRef.setInput('hasPreviousFrame', false);
    open(makeFrame('f0', ['top', 'side', 'front'], { unlabeled: true }));
    expect(shownImage()).toBe('f0_top.png');
    expect(actionButton('← Back').classList).toContain('btn-disabled');
  });

  it('Clear values empties the selected view only, without saving', () => {
    const frame = makeFrame('f1', ['top', 'side']);
    open(frame);
    const clear = (): HTMLButtonElement =>
      fixture.nativeElement.querySelector(
        'button[aria-label="Clear values in this view"]',
      );
    expect(clear().disabled).toBeFalse();

    clear().click();
    fixture.detectChanges();
    expect(frame.views[0].keypoints.every((kp) => isNaN(kp.x))).toBeTrue();
    expect(frame.views[1].keypoints.every((kp) => !isNaN(kp.x))).toBeTrue();
    expect(clear().disabled).toBeTrue();
    expect(actionButton('Revert changes').classList).not.toContain(
      'btn-disabled',
    );
    expect(session.saveMVFrame).not.toHaveBeenCalled();
  });

  it('selects the first keypoint on a new view unless Keep selected keypoint is on', () => {
    open(makeFrame('f1', ['top', 'side', 'front']));
    selectKeypoint('tail');
    clickNext();
    expect(selectedKeypoint()).toBe('nose');

    const keep: HTMLInputElement = fixture.nativeElement.querySelector(
      'input[aria-label="Keep selected keypoint"]',
    );
    expect(keep.checked).toBeFalse();
    keep.click();
    fixture.detectChanges();
    selectKeypoint('tail');
    clickNext();
    expect(shownImage()).toBe('f1_front.png');
    expect(selectedKeypoint()).toBe('tail');

    // Also on the next frame, including one from the remaining list.
    clickNext();
    open(makeFrame('f2', ['top', 'side', 'front'], { unlabeled: true }));
    expect(shownImage()).toBe('f2_top.png');
    expect(selectedKeypoint()).toBe('tail');
  });

  it('keeps the open view when the frame is replaced by a reloaded copy (Refresh)', () => {
    open(makeFrame('f1', ['top', 'side']));
    clickNext();
    expect(shownImage()).toBe('f1_side.png');

    // Refresh: the page shows no frame while loading, then a new copy.
    open(null as unknown as MVFrame);
    open(makeFrame('f1', ['top', 'side']));
    expect(shownImage()).toBe('f1_side.png');
  });

  it('placing a keypoint selects the next unlabeled one unless Keep selected keypoint is on', () => {
    open(makeFrame('f1', ['top'], { unlabeled: true }));
    const widget = fixture.debugElement.query(
      By.directive(ImageLabelWidgetStubComponent),
    ).componentInstance as ImageLabelWidgetStubComponent;
    const place = (kp: string) => {
      widget.keypointUpdated.emit({ kp, position: { x: 1, y: 2 } });
      fixture.detectChanges();
    };

    expect(selectedKeypoint()).toBe('nose');
    place('nose');
    expect(selectedKeypoint()).toBe('ear');

    const keep: HTMLInputElement = fixture.nativeElement.querySelector(
      'input[aria-label="Keep selected keypoint"]',
    );
    keep.click();
    fixture.detectChanges();
    place('ear');
    expect(selectedKeypoint()).toBe('ear');
  });
});

describe('LabelerCenterPanelComponent label audit review', () => {
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;
  const VIEWS = ['top', 'side'];
  const NAMES = ['nose', 'leftEar', 'rightEar'];

  const frame: MVFrame = {
    key: 'labeled-data/s_*/img01.png',
    views: VIEWS.map((viewName) => {
      const keypoints = NAMES.map((keypointName, i) => ({ keypointName, x: i, y: i }));
      return {
        viewName,
        imgPath: `labeled-data/s_${viewName}/img01.png`,
        keypoints,
        originalKeypoints: keypoints,
      };
    }),
  };

  function focus(index = 1, partner: string | null = null): AuditFocus {
    return {
      index,
      total: 3,
      finding: {
        kind: partner ? 'swap' : 'multiview',
        view: 'side',
        keypoint: 'leftEar',
        partner,
        imgPaths: { top: frame.views[0].imgPath, side: frame.views[1].imgPath },
        errorPx: 20,
        tolerancePx: 5,
        score: 4,
        reason: 'leftEar in side is 20px from where top put it',
        frameKey: frame.key,
      },
    };
  }

  beforeEach(async () => {
    const projectInfo = jasmine.createSpyObj(
      'ProjectInfoService',
      ['projectContext'],
      { projectInfo: { data_dir: 'data', views: VIEWS } },
    );
    projectInfo.projectContext.and.returnValue(null);
    const calibration = jasmine.createSpyObj('CameraCalibrationService', [
      'getCalibrationStatus',
    ]);
    calibration.getCalibrationStatus.and.resolveTo('none');

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibration },
        { provide: SessionService, useValue: jasmine.createSpyObj('SessionService', ['saveMVFrame']) },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: VIEWS.map((v) => ({ viewName: v, csvPath: `CollectedData_${v}.csv` })),
    });
    fixture.componentRef.setInput('frame', frame);
  });

  function reviewBar(): HTMLElement | null {
    return fixture.nativeElement.querySelector('[data-testid="audit-review-bar"]');
  }

  function mainWidget(): ImageLabelWidgetStubComponent {
    return fixture.debugElement.queryAll(By.directive(ImageLabelWidgetStubComponent))[0]
      .componentInstance;
  }

  it('shows no review bar outside a review', () => {
    fixture.detectChanges();
    expect(reviewBar()).toBeNull();
  });

  it('opens the flagged view with the flagged keypoint selected and says why', () => {
    fixture.componentRef.setInput('auditFocus', focus());
    fixture.detectChanges();

    expect(reviewBar()!.textContent).toContain('Likely mistake 2 / 3');
    expect(reviewBar()!.textContent).toContain('20px from where top put it');
    expect((mainWidget().fv() as { viewName: string }).viewName).toBe('side');
    expect(mainWidget().selectedKeypoint()).toBe('leftEar');
  });

  it('draws only the flagged keypoints in the main image', () => {
    fixture.componentRef.setInput('auditFocus', focus(1, 'rightEar'));
    fixture.componentRef.setInput('visibleKeypoints', ['leftEar', 'rightEar']);
    fixture.detectChanges();

    expect(mainWidget().visibleKeypoints()).toEqual(['leftEar', 'rightEar']);
  });

  it('has no step buttons of its own (Back / Next / Save and next step); Esc stops', () => {
    const exit = jasmine.createSpy('exit');
    fixture.componentInstance.auditExit.subscribe(exit);
    fixture.componentRef.setInput('auditFocus', focus());
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('button[aria-label="Next likely mistake"]')).toBeNull();
    expect(fixture.nativeElement.querySelector('button[aria-label="Previous likely mistake"]')).toBeNull();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'n' }));
    expect(exit).not.toHaveBeenCalled();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(exit).toHaveBeenCalledTimes(1);
  });

  it('Stop review removes the review bar, the lines and the keypoint filter', () => {
    fixture.componentRef.setInput('auditFocus', focus());
    fixture.componentRef.setInput('visibleKeypoints', ['leftEar']);
    fixture.detectChanges();
    expect(reviewBar()).not.toBeNull();

    // The page answers Stop review by clearing the focus and the filter.
    fixture.componentRef.setInput('auditFocus', null);
    fixture.componentRef.setInput('visibleKeypoints', null);
    fixture.detectChanges();

    expect(reviewBar()).toBeNull();
    expect(mainWidget().epipolarGuides()).toBeNull();
    expect(mainWidget().visibleKeypoints()).toBeNull();
  });

});

describe('LabelerCenterPanelComponent distances of a rigid group in the review', () => {
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;
  let auditService: jasmine.SpyObj<LabelAuditService>;
  const VIEWS = ['top', 'side'];
  const NAMES = ['nose', 'leftEar', 'rightEar'];

  /** A new frame each time: edits change the frame object. */
  function frameOf(img: string): MVFrame {
    return {
      key: `labeled-data/s_*/${img}.png`,
      views: VIEWS.map((viewName) => {
        const keypoints = NAMES.map((keypointName, i) => ({ keypointName, x: 10 * i, y: 10 * i }));
        return {
          viewName,
          imgPath: `labeled-data/s_${viewName}/${img}.png`,
          keypoints,
          originalKeypoints: keypoints,
        };
      }),
    };
  }

  /** The pairs of the group with leftEar to nose at `percent` of its usual distance (10). */
  function pairs(percent: number): ShapePair[] {
    const pair = (a: string, b: string, distance: number): ShapePair => ({
      a,
      b,
      usual: 10,
      spread: 0.5,
      limit: 1.5,
      frames: 50,
      distance,
      off: Math.abs(distance - 10) > 1.5,
    });
    return [pair('nose', 'leftEar', percent / 10), pair('nose', 'rightEar', 10), pair('leftEar', 'rightEar', 10)];
  }

  function focusOn(frame: MVFrame, index: number, percent: number): AuditFocus {
    return {
      index,
      total: 3,
      finding: {
        kind: 'shape',
        view: 'top',
        keypoint: 'leftEar',
        partner: null,
        imgPaths: { top: frame.views[0].imgPath, side: frame.views[1].imgPath },
        errorPx: 5,
        tolerancePx: 1.5,
        score: 3,
        reason: 'leftEar to nose too long',
        group: NAMES,
        groupPairs: pairs(percent),
        frameKey: frame.key,
      },
    };
  }

  beforeEach(async () => {
    const projectInfo = jasmine.createSpyObj('ProjectInfoService', ['projectContext'], {
      projectInfo: { data_dir: 'data', views: VIEWS },
    });
    projectInfo.projectContext.and.returnValue(null);
    const calibration = jasmine.createSpyObj('CameraCalibrationService', ['getCalibrationStatus']);
    calibration.getCalibrationStatus.and.resolveTo('none');
    auditService = jasmine.createSpyObj('LabelAuditService', ['shapeGuides', 'midlineGuides']);
    auditService.shapeGuides.and.resolveTo({ group: NAMES, groupPairs: pairs(100) });

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibration },
        { provide: SessionService, useValue: jasmine.createSpyObj('SessionService', ['saveMVFrame']) },
        { provide: LabelAuditService, useValue: auditService },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: VIEWS.map((v) => ({ viewName: v, csvPath: `CollectedData_${v}.csv` })),
    });
  });

  function mainWidget(): ImageLabelWidgetStubComponent {
    return fixture.debugElement.queryAll(By.directive(ImageLabelWidgetStubComponent))[0]
      .componentInstance;
  }

  /** The distances in the review bar, or null when none are shown. */
  function distances(): string | null {
    const el = fixture.nativeElement.querySelector('[data-testid="audit-shape-reason"]');
    return el ? (el.textContent as string).replace(/\s+/g, ' ').trim() : null;
  }

  /** Opens a finding as the page does: its frame and the focus. */
  function open(frame: MVFrame, index: number, percent: number) {
    fixture.componentRef.setInput('frame', frame);
    fixture.componentRef.setInput('auditFocus', focusOn(frame, index, percent));
    fixture.detectChanges();
  }

  /** Lets the measurement after an edit come back and be drawn. */
  async function settle() {
    fixture.detectChanges();
    // Twice: the request starts in a timer, its answer arrives in a later task.
    for (let i = 0; i < 2; i++) {
      await new Promise((resolve) => setTimeout(resolve, 5));
      fixture.detectChanges();
    }
  }

  /** The user drags leftEar in the shown view. */
  async function moveLeftEar() {
    mainWidget().keypointUpdated.emit({ kp: 'leftEar', position: { x: 3, y: 4 } });
    await settle();
  }

  it('shows the distances of the check when a finding opens, without measuring', () => {
    open(frameOf('img01'), 0, 150);

    expect(distances()).toContain('leftEar to nose too long (150% of usual)');
    expect(auditService.shapeGuides).not.toHaveBeenCalled();
  });

  it('measures again after an edit, with the frame on screen, and shows the new distances', async () => {
    const frame = frameOf('img01');
    open(frame, 0, 150);

    await moveLeftEar();

    expect(auditService.shapeGuides).toHaveBeenCalledOnceWith(frame, 'leftEar');
    expect(distances()).toContain('leftEar: all distances as usual');
  });

  it('shows the distances of the next finding, not the measurement of the one before', async () => {
    open(frameOf('img01'), 0, 150);
    await moveLeftEar();

    open(frameOf('img02'), 1, 200);

    expect(distances()).toContain('leftEar to nose too long (200% of usual)');
  });

  it('shows the distances of the next finding when it is in the same frame', async () => {
    const frame = frameOf('img01');
    open(frame, 0, 150);
    await moveLeftEar();

    open(frame, 1, 50);

    expect(distances()).toContain('leftEar to nose too short (50% of usual)');
  });

  it('measures the frame on screen after stepping to another finding', async () => {
    open(frameOf('img01'), 0, 150);
    await moveLeftEar();
    const second = frameOf('img02');
    open(second, 1, 200);
    auditService.shapeGuides.calls.reset();

    await moveLeftEar();

    expect(auditService.shapeGuides).toHaveBeenCalledOnceWith(second, 'leftEar');
    expect(distances()).toContain('leftEar: all distances as usual');
  });

  it('going back to a finding shows its last measurement, not the numbers of the check', async () => {
    const first = frameOf('img01');
    open(first, 0, 150);
    await moveLeftEar(); // as usual now
    open(frameOf('img02'), 1, 200);
    auditService.shapeGuides.and.resolveTo({ group: NAMES, groupPairs: pairs(50) });
    await moveLeftEar();
    expect(distances()).toContain('too short (50% of usual)');

    open(first, 0, 150);

    expect(distances()).toContain('leftEar: all distances as usual');
  });

  it('a late answer for the finding before does not change the one on screen', async () => {
    let answer: (guides: ShapeGuides) => void = () => undefined;
    auditService.shapeGuides.and.returnValue(new Promise((resolve) => (answer = resolve)));
    open(frameOf('img01'), 0, 150);
    await moveLeftEar(); // asked, not answered yet

    open(frameOf('img02'), 1, 200);
    answer({ group: NAMES, groupPairs: pairs(100) });
    await settle();

    expect(distances()).toContain('leftEar to nose too long (200% of usual)');
    expect(fixture.nativeElement.querySelector('[data-testid="shape-refit-state"]')).toBeNull();
  });

  it('shows no distances of the finding while another frame is on screen', () => {
    const first = frameOf('img01');
    open(first, 0, 150);

    // The frame changed, the finding has not followed yet.
    fixture.componentRef.setInput('frame', frameOf('img02'));
    fixture.detectChanges();

    expect(distances()).toBeNull();
    expect(mainWidget().shapeLines()).toEqual([]);
  });

  it('shows the distances of the keypoint that is selected', () => {
    open(frameOf('img01'), 0, 150);

    mainWidget().selectedKeypoint.set('rightEar');
    fixture.detectChanges();

    expect(distances()).toContain('rightEar: all distances as usual');
  });

  it('draws a line per measured pair and marks the ones that are off', async () => {
    open(frameOf('img01'), 0, 150);
    expect(mainWidget().shapeLines().map((l) => [l.name, l.off])).toEqual([
      ['nose|leftEar', 'long'],
      ['nose|rightEar', null],
      ['leftEar|rightEar', null],
    ]);

    await moveLeftEar();

    expect(mainWidget().shapeLines().map((l) => l.off)).toEqual([null, null, null]);
  });

  it('forgets the measurements when the review stops', async () => {
    const frame = frameOf('img01');
    open(frame, 0, 150);
    await moveLeftEar();
    expect(distances()).toContain('all distances as usual');

    fixture.componentRef.setInput('auditFocus', null);
    fixture.detectChanges();
    expect(distances()).toBeNull();
    // A new check numbers its findings from 0 again.
    open(frame, 0, 150);

    expect(distances()).toContain('leftEar to nose too long (150% of usual)');
  });
});

describe('LabelerCenterPanelComponent epipolar lines while labeling', () => {
  let fixture: ComponentFixture<LabelerCenterPanelComponent>;
  let rpc: jasmine.SpyObj<RpcService>;
  let viewOptions: LabelerViewOptionsService;
  const VIEWS = ['top', 'side'];
  const GUIDES = {
    lines: [{ fromView: 'side', segments: [[{ x: 0, y: 0 }, { x: 10, y: 0 }]], rank: 0, crossingAngleDeg: null }],
    predictedPoint: null,
    predictedFromViews: [],
  };

  /** `saved`: a labeled frame; else a frame of the queue, with its predictions. */
  function frameOf(saved: boolean): MVFrame {
    return {
      key: 'labeled-data/s_*/img01.png',
      views: VIEWS.map((viewName) => {
        const keypoints = [{ keypointName: 'nose', x: 3, y: 4 }];
        return {
          viewName,
          imgPath: `labeled-data/s_${viewName}/img01.png`,
          keypoints,
          originalKeypoints: saved ? keypoints : null,
        };
      }),
    };
  }

  async function setUp(calibration: 'session' | 'none', on: boolean) {
    localStorage.setItem('labeler-view-options.showEpipolarLinesWhileLabeling', String(on));
    const projectInfo = jasmine.createSpyObj('ProjectInfoService', ['projectContext'], {
      projectInfo: { data_dir: 'data', views: VIEWS },
    });
    projectInfo.projectContext.and.returnValue({ key: 'proj', projectInfo: { views: VIEWS } });
    const calibrationService = jasmine.createSpyObj('CameraCalibrationService', ['getCalibrationStatus']);
    calibrationService.getCalibrationStatus.and.resolveTo(calibration);
    rpc = jasmine.createSpyObj('RpcService', ['call']);
    rpc.call.and.resolveTo(GUIDES);

    await TestBed.configureTestingModule({
      imports: [LabelerCenterPanelComponent],
      providers: [
        { provide: ProjectInfoService, useValue: projectInfo },
        { provide: CameraCalibrationService, useValue: calibrationService },
        { provide: SessionService, useValue: jasmine.createSpyObj('SessionService', ['saveMVFrame']) },
        { provide: RpcService, useValue: rpc },
      ],
    })
      .overrideComponent(LabelerCenterPanelComponent, {
        remove: { imports: [ImageLabelWidgetComponent] },
        add: { imports: [ImageLabelWidgetStubComponent] },
      })
      .compileComponents();

    fixture = TestBed.createComponent(LabelerCenterPanelComponent);
    // Provided by the component, not the root.
    viewOptions = fixture.debugElement.injector.get(LabelerViewOptionsService);
    fixture.componentRef.setInput('numLabeledFramesGetter', () => 0);
    fixture.componentRef.setInput('labelFile', {
      key: 'CollectedData',
      views: VIEWS.map((v) => ({ viewName: v, csvPath: `CollectedData_${v}.csv` })),
    });
  }

  afterEach(() => localStorage.removeItem('labeler-view-options.showEpipolarLinesWhileLabeling'));

  async function show(frame: MVFrame) {
    fixture.componentRef.setInput('frame', frame);
    fixture.detectChanges();
    // The user clicks the keypoint (a saved frame opens with none selected).
    mainWidget().selectedKeypoint.set('nose');
    for (let i = 0; i < 3; i++) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    fixture.detectChanges();
  }

  function mainWidget(): ImageLabelWidgetStubComponent {
    return fixture.debugElement.queryAll(By.directive(ImageLabelWidgetStubComponent))[0]
      .componentInstance;
  }

  function toggle(): HTMLInputElement | null {
    return fixture.nativeElement.querySelector('input[aria-label="Show epipolar lines while labeling"]');
  }

  function guideCalls() {
    return rpc.call.calls.all().filter((c) => c.args[0] === 'getEpipolarGuides');
  }

  it('draws no lines while labeling until they are switched on', async () => {
    await setUp('session', false);
    await show(frameOf(true));

    expect(toggle()!.checked).toBeFalse();
    expect(mainWidget().epipolarGuides()).toBeNull();
    expect(guideCalls().length).toBe(0);
  });

  it('draws the lines of the selected keypoint on a saved frame, outside a review', async () => {
    await setUp('session', true);
    await show(frameOf(true));

    expect(guideCalls().length).toBeGreaterThan(0);
    expect(guideCalls()[0].args[1]).toEqual(jasmine.objectContaining({ sessionKey: 's', targetView: 'top' }));
    expect(mainWidget().epipolarGuides()).toEqual(GUIDES);
  });

  it('draws them on a frame of the queue too', async () => {
    await setUp('session', true);
    await show(frameOf(false));

    expect(mainWidget().epipolarGuides()).toEqual(GUIDES);
  });

  it('says how far the label is from the line, without a tolerance', async () => {
    await setUp('session', true);
    await show(frameOf(true));

    const distance = fixture.nativeElement.querySelector('[data-testid="line-distance"]');
    expect(distance.textContent).toContain('4 px from the side line');
    expect(distance.textContent).not.toContain('✓');
  });

  it('the switch turns the lines on and off', async () => {
    await setUp('session', false);
    await show(frameOf(true));

    toggle()!.click();
    await show(frameOf(true));
    expect(viewOptions.showEpipolarLinesWhileLabeling()).toBeTrue();
    expect(mainWidget().epipolarGuides()).toEqual(GUIDES);

    toggle()!.click();
    await show(frameOf(true));
    expect(mainWidget().epipolarGuides()).toBeNull();
  });

  it('has no switch and no lines without a camera calibration', async () => {
    await setUp('none', true);
    await show(frameOf(true));

    expect(toggle()).toBeNull();
    expect(mainWidget().epipolarGuides()).toBeNull();
    expect(guideCalls().length).toBe(0);
  });

  it('in a review the review\'s own switch decides, not this one', async () => {
    await setUp('session', false);
    const frame = frameOf(true);
    fixture.componentRef.setInput('auditFocus', {
      index: 0,
      total: 1,
      finding: {
        kind: 'multiview', view: 'top', keypoint: 'nose', partner: null,
        imgPaths: { top: frame.views[0].imgPath, side: frame.views[1].imgPath },
        errorPx: 20, tolerancePx: 5, score: 4, reason: 'off', frameKey: frame.key,
      },
    });
    await show(frame);

    expect(toggle()).toBeNull();
    expect(mainWidget().epipolarGuides()).toEqual(GUIDES);
  });
});

@Component({ selector: 'app-image-label-widget', template: '' })
class ImageLabelWidgetStubComponent {
  fv = input();
  revision = input();
  imageReload = input();
  selectedKeypoint = model<string | null>(null);
  disableInteractions = input();
  zoomableInteractivityDisabled = input();
  visibleKeypoints = input<string[] | null>(null);
  epipolarGuides = input<unknown>(null);
  shapeLines = input<{ name: string; off: string | null; main: boolean }[]>([]);
  midlineGuide = input<unknown>(null);
  guideViews = input<string[]>([]);
  keypointUpdated = output<unknown>();
  zoomableContent = () => null;
  imageSize = signal<{ w: number; h: number } | null>(null);
  contextFramesMissing = signal(false);
  currentFrameMissing = signal(false);
  currentFrameMessage = signal<string | null>(null);
  contextSlotViews = signal<ContextSlotView[] | null>(null);
}
