import { TestBed } from '@angular/core/testing';
import {
  LabelAuditService,
  matchFindingsToFrames,
  missingFindings,
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
      midline: null,
      scope: 'labeled',
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

  it('sends the midline keypoints, guessed from the names until the user chooses', async () => {
    const kp = (keypointName: string) => ({ keypointName, x: 1, y: 1 });
    const f = frame(1);
    f.views.forEach(
      (v) => (v.keypoints = ['nose', 'leftShoulder', 'rightShoulder', 'back', 'tail'].map(kp) as never),
    );
    rpc.call.and.resolveTo({ items: [], framesChecked: 1, pointsChecked: 5, sessionsWithoutCalibration: [] });

    await service.run(labelFile, [f]);
    const sent = () => (rpc.call.calls.mostRecent().args[1] as { midline: unknown }).midline;
    expect(sent()).toEqual({ left: 'leftShoulder', right: 'rightShoulder', point: 'back', toleranceDeg: 5 });

    service.setMidline({ point: 'tail', toleranceDeg: 8 });
    await service.run(labelFile, [f]);
    expect(sent()).toEqual({ left: 'leftShoulder', right: 'rightShoulder', point: 'tail', toleranceDeg: 8 });
    localStorage.removeItem('label-audit.midline.proj');
  });

  it('reviews midline findings on their own tab, worst first, with the left and right keypoints shown', async () => {
    const midline = (n: number, score: number) =>
      item(n, { kind: 'midline', keypoint: 'back', score, group: ['back', 'leftShoulder', 'rightShoulder'] });
    await runWith([item(1), midline(1, 1.5), midline(2, 4)]);

    expect(service.findings().map((f) => f.kind)).toEqual(['multiview']);
    expect(service.midlineCount()).toBe(2);

    service.setMode('midline');
    expect(service.findings().map((f) => f.score)).toEqual([4, 1.5]);
    service.startReview();
    expect(service.focus()?.finding.frameKey).toBe(frame(2).key);
    expect(service.visibleKeypoints()).toEqual(['back', 'leftShoulder', 'rightShoulder']);
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

describe('Label audit, missing keypoint', () => {
  /** A frame with nose in both views and leftEar only in the views named. */
  function frameWith(n: number, leftEarIn: string[]): MVFrame {
    return {
      key: `labeled-data/s_*/img${n}.png`,
      views: ['top', 'side'].map((viewName) => ({
        viewName,
        imgPath: `labeled-data/s_${viewName}/img${n}.png`,
        keypoints: [
          { keypointName: 'nose', x: 1, y: 1 },
          { keypointName: 'leftEar', x: leftEarIn.includes(viewName) ? 2 : NaN, y: leftEarIn.includes(viewName) ? 2 : NaN },
        ],
        originalKeypoints: [],
      })),
    };
  }
  const frames = [frameWith(1, ['top', 'side']), frameWith(2, ['top']), frameWith(3, []), frameWith(4, ['side'])];
  const names = (where: 'any' | 'all' | 'some', keypoint = 'leftEar') =>
    missingFindings(frames, keypoint, where).map((f) => f.frameKey.slice(-8, -4));

  it('lists the frames where the keypoint has no label in at least one view, in frame order', () => {
    expect(names('any')).toEqual(['img2', 'img3', 'img4']);
  });

  it('can list only the frames without it in every view, or in some views only', () => {
    expect(names('all')).toEqual(['img3']);
    expect(names('some')).toEqual(['img2', 'img4']);
  });

  it('lists nothing for a keypoint that is labeled everywhere, or when none is chosen', () => {
    expect(names('any', 'nose')).toEqual([]);
    expect(names('any', '')).toEqual([]);
  });

  it('says in which view it is missing, and opens that view', () => {
    const [second, third] = missingFindings(frames, 'leftEar', 'any');
    expect([second.view, second.reason]).toEqual(['side', 'leftEar is missing in side']);
    expect(third.reason).toBe('leftEar is missing in every view');
    expect(second.imgPaths).toEqual({ top: 'labeled-data/s_top/img2.png', side: 'labeled-data/s_side/img2.png' });
  });

  describe('in the service', () => {
    let service: LabelAuditService;
    let rpc: jasmine.SpyObj<RpcService>;
    const labelFile = {
      key: 'CollectedData',
      views: [
        { viewName: 'top', csvPath: '/d/CollectedData_top.csv' },
        { viewName: 'side', csvPath: '/d/CollectedData_side.csv' },
      ],
    };
    const swap: LabelAuditItem = {
      kind: 'swap',
      view: 'side',
      keypoint: 'leftEar',
      partner: 'nose',
      imgPaths: { top: 'labeled-data/s_top/img1.png', side: 'labeled-data/s_side/img1.png' },
      errorPx: 40,
      tolerancePx: 5,
      score: 8,
      reason: 'swapped',
    };

    beforeEach(() => {
      localStorage.removeItem('label-audit.missing-keypoint.proj');
      localStorage.removeItem('label-audit.missing-where.proj');
      rpc = jasmine.createSpyObj('RpcService', ['call']);
      rpc.call.and.resolveTo({ items: [swap], framesChecked: 4, pointsChecked: 8, sessionsWithoutCalibration: [] });
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

    afterEach(() => {
      localStorage.removeItem('label-audit.missing-keypoint.proj');
      localStorage.removeItem('label-audit.missing-where.proj');
    });

    it('lists nothing as missing until a keypoint is chosen', async () => {
      await service.run(labelFile, frames);
      service.setMode('missing');

      expect(service.findings()).toEqual([]);
      expect(service.viewsCount()).toBe(1);
    });

    it('has its own list next to the other audits', async () => {
      service.setMissing({ keypoint: 'leftEar' });
      await service.run(labelFile, frames);

      expect([service.viewsCount(), service.missingCount()]).toEqual([1, 3]);
      expect(service.findings().map((f) => f.kind)).toEqual(['swap']);
      service.setMode('missing');
      expect(service.findings().map((f) => f.kind)).toEqual(['missing', 'missing', 'missing']);
    });

    it('follows another keypoint or view rule at once, without a new check, and stops the review', async () => {
      service.setMissing({ keypoint: 'leftEar' });
      await service.run(labelFile, frames);
      service.setMode('missing');
      service.startReview(1);

      service.setMissing({ where: 'all' });

      expect(rpc.call).toHaveBeenCalledTimes(1);
      expect(service.isReviewing()).toBeFalse();
      expect(service.findings().map((f) => f.frameKey)).toEqual([frames[2].key]);
      service.setMissing({ keypoint: 'nose' });
      expect(service.findings()).toEqual([]);
      expect(service.viewsCount()).toBe(1);
    });

    it('draws all keypoints while a missing one is reviewed', async () => {
      service.setMissing({ keypoint: 'leftEar' });
      await service.run(labelFile, frames);
      service.setMode('missing');

      service.startReview();

      expect(service.focus()?.finding.keypoint).toBe('leftEar');
      expect(service.visibleKeypoints()).toBeNull();
    });

    it('remembers the keypoint for the project', () => {
      service.setMissing({ keypoint: 'leftEar', where: 'some' });

      const again = TestBed.runInInjectionContext(() => new LabelAuditService());

      expect([again.missingKeypoint(), again.missingWhere()]).toEqual(['leftEar', 'some']);
    });
  });
});
