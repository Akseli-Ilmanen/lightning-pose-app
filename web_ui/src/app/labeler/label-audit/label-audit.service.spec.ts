import { TestBed } from '@angular/core/testing';
import {
  LabelAuditService,
  matchFindingsToFrames,
  rigidGroupsOf,
} from './label-audit.service';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { MVFrame } from '../frame.model';
import { LabelAuditItem } from './label-audit.model';

function frame(n: number): MVFrame {
  return {
    key: `labeled-data/s_*/img0${n}.png`,
    views: ['top', 'side'].map((viewName) => ({
      viewName,
      imgPath: `labeled-data/s_${viewName}/img0${n}.png`,
      keypoints: [],
      originalKeypoints: [],
    })),
  };
}

function item(n: number, over: Partial<LabelAuditItem> = {}): LabelAuditItem {
  return {
    kind: 'multiview',
    view: 'side',
    keypoint: 'nose',
    partner: null,
    imgPaths: { top: `labeled-data/s_top/img0${n}.png`, side: `labeled-data/s_side/img0${n}.png` },
    errorPx: 12,
    tolerancePx: 5,
    score: 2.4,
    reason: 'r',
    ...over,
  };
}

describe('matchFindingsToFrames', () => {
  it('finds the frame of each item by the image path of its view', () => {
    const out = matchFindingsToFrames([item(2), item(1)], [frame(1), frame(2)]);
    expect(out.map((f) => f.frameKey)).toEqual([frame(2).key, frame(1).key]);
  });

  it('drops items whose frame is not loaded', () => {
    expect(matchFindingsToFrames([item(9)], [frame(1)])).toEqual([]);
  });
});

describe('LabelAuditService', () => {
  let service: LabelAuditService;
  let rpc: jasmine.SpyObj<RpcService>;

  beforeEach(() => {
    localStorage.removeItem('label-audit.keypoints.proj');
    rpc = jasmine.createSpyObj('RpcService', ['call']);
    const projectInfo = jasmine.createSpyObj('ProjectInfoService', ['projectContext']);
    projectInfo.projectContext.and.returnValue({ key: 'proj' });
    TestBed.configureTestingModule({
      providers: [
        LabelAuditService,
        { provide: RpcService, useValue: rpc },
        { provide: ProjectInfoService, useValue: projectInfo },
      ],
    });
    service = TestBed.inject(LabelAuditService);
  });

  const labelFile = {
    key: 'CollectedData',
    views: [
      { viewName: 'top', csvPath: '/d/CollectedData_top.csv' },
      { viewName: 'side', csvPath: '/d/CollectedData_side.csv' },
    ],
  };

  async function runWith(items: LabelAuditItem[]) {
    rpc.call.and.resolveTo({
      items,
      framesChecked: 2,
      pointsChecked: 6,
      sessionsWithoutCalibration: [],
    });
    await service.run(labelFile, [frame(1), frame(2)]);
  }

  it('sends every view and the tolerance', async () => {
    service.tolerancePx.set(3);
    await runWith([]);
    expect(rpc.call).toHaveBeenCalledWith('getLabelAudit', {
      projectKey: 'proj',
      views: [
        { viewName: 'top', csvPath: '/d/CollectedData_top.csv' },
        { viewName: 'side', csvPath: '/d/CollectedData_side.csv' },
      ],
      defaultTolerancePx: 3,
      rigidGroups: [],
      keypoints: null,
      shapeStrictness: 3,
      animals: [],
    });
    expect(service.auditedLabelFileKey()).toBe('CollectedData');
    expect(service.summary()?.framesChecked).toBe(2);
  });

  it('sends the ticked keypoints and the rigid groups, and remembers them for the project', async () => {
    const kp = (keypointName: string) => ({ keypointName, x: 1, y: 1 });
    const f = frame(1);
    f.views.forEach((v) => (v.keypoints = ['nose', 'earL', 'earR', 'top', 'tail'].map(kp) as never));
    ['nose', 'earL', 'earR', 'top'].forEach((n) => service.setGroup(n, 1));
    service.setChecked(['tail'], false);
    rpc.call.and.resolveTo({ items: [], framesChecked: 1, pointsChecked: 5, sessionsWithoutCalibration: [] });

    await service.run(labelFile, [f]);

    const payload = rpc.call.calls.mostRecent().args[1] as { rigidGroups: string[][]; keypoints: string[] };
    expect(payload.rigidGroups).toEqual([['nose', 'earL', 'earR', 'top']]);
    expect(payload.keypoints).toEqual(['nose', 'earL', 'earR', 'top']);
    expect(JSON.parse(localStorage.getItem('label-audit.keypoints.proj')!)).toEqual({
      unchecked: ['tail'],
      groups: { nose: 1, earL: 1, earR: 1, top: 1 },
    });
    localStorage.removeItem('label-audit.keypoints.proj');
  });

  it('reviews findings in order and stops at the ends', async () => {
    await runWith([item(1), item(2)]);
    service.startReview();
    expect(service.focus()?.index).toBe(0);
    service.previous();
    expect(service.focus()?.index).toBe(0);
    service.next();
    service.next();
    expect(service.focus()?.index).toBe(1);
    expect(service.focus()?.total).toBe(2);
    service.stopReview();
    expect(service.focus()).toBeNull();
  });

  it('shows only the flagged keypoint and its swap partner while reviewing', async () => {
    await runWith([item(1, { kind: 'swap', keypoint: 'leftEar', partner: 'rightEar' })]);
    expect(service.visibleKeypoints()).toBeNull();
    service.startReview();
    expect(service.visibleKeypoints()).toEqual(['leftEar', 'rightEar']);
    service.onlyFlaggedKeypoints.set(false);
    expect(service.visibleKeypoints()).toBeNull();
  });

  it('shows the server error and keeps no findings', async () => {
    rpc.call.and.rejectWith({ error: { detail: 'needs two views' } });
    await service.run(labelFile, []);
    expect(service.error()).toBe('needs two views');
    expect(service.findings()).toEqual([]);
    expect(service.isRunning()).toBeFalse();
  });
});

describe('rigidGroupsOf', () => {
  it('has no group when no keypoint has a number', () => {
    expect(rigidGroupsOf({}, ['a', 'b'])).toEqual([]);
  });

  it('lists each group in number order, names in keypoint order', () => {
    expect(rigidGroupsOf({ c: 2, a: 1, b: 1, d: 2, x: 0 }, ['a', 'b', 'c', 'd', 'x'])).toEqual([
      ['a', 'b'],
      ['c', 'd'],
    ]);
  });
});
