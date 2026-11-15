import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LabelAuditPanelComponent } from './label-audit-panel.component';
import { LabelAuditService } from './label-audit.service';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { MVFrame } from '../frame.model';

describe('LabelAuditPanelComponent', () => {
  let fixture: ComponentFixture<LabelAuditPanelComponent>;
  let rpc: jasmine.SpyObj<RpcService>;
  let audit: LabelAuditService;

  const frame: MVFrame = {
    key: 'labeled-data/s_*/img01.png',
    views: ['top', 'side'].map((viewName) => ({
      viewName,
      imgPath: `labeled-data/s_${viewName}/img01.png`,
      keypoints: [],
      originalKeypoints: [],
    })),
  };
  const labelFile = {
    key: 'CollectedData',
    views: [
      { viewName: 'top', csvPath: 'CollectedData_top.csv' },
      { viewName: 'side', csvPath: 'CollectedData_side.csv' },
    ],
  };

  beforeEach(async () => {
    rpc = jasmine.createSpyObj('RpcService', ['call']);
    const projectInfo = jasmine.createSpyObj('ProjectInfoService', ['projectContext']);
    projectInfo.projectContext.and.returnValue({ key: 'proj' });
    await TestBed.configureTestingModule({
      imports: [LabelAuditPanelComponent],
      providers: [
        LabelAuditService,
        { provide: RpcService, useValue: rpc },
        { provide: ProjectInfoService, useValue: projectInfo },
      ],
    }).compileComponents();
    audit = TestBed.inject(LabelAuditService);
    fixture = TestBed.createComponent(LabelAuditPanelComponent);
    fixture.componentRef.setInput('labelFile', labelFile);
    fixture.componentRef.setInput('frames', [frame]);
    fixture.detectChanges();
  });

  function checkButton(): HTMLButtonElement {
    return Array.from(fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>)
      .find((b) => b.textContent!.includes('Check labels'))!;
  }

  async function check(items: unknown[]) {
    rpc.call.and.resolveTo({ items, framesChecked: 1, pointsChecked: 3, sessionsWithoutCalibration: ['s9'] });
    checkButton().click();
    await fixture.whenStable();
    fixture.detectChanges();
  }

  it('cannot check a single-view label file', () => {
    fixture.componentRef.setInput('labelFile', { key: 'x', views: [labelFile.views[0]] });
    fixture.detectChanges();
    expect(checkButton().disabled).toBeTrue();
  });

  it('lists the likely mistakes and opens one for review when clicked', async () => {
    await check([
      {
        kind: 'swap',
        view: 'side',
        keypoint: 'leftEar',
        partner: 'rightEar',
        imgPaths: { top: frame.views[0].imgPath, side: frame.views[1].imgPath },
        errorPx: 40,
        tolerancePx: 5,
        score: 8,
        reason: 'leftEar and rightEar look swapped in side',
      },
    ]);

    const options = fixture.nativeElement.querySelectorAll('[role="option"] button');
    expect(options.length).toBe(1);
    expect(options[0].textContent).toContain('side · leftEar/rightEar');
    expect(fixture.nativeElement.textContent).toContain('Not checked, no calibration: s9');

    options[0].click();
    fixture.detectChanges();
    expect(audit.focus()?.finding.keypoint).toBe('leftEar');
  });

  it('says so when nothing is found', async () => {
    await check([]);
    expect(fixture.nativeElement.querySelector('[data-testid="audit-summary"]').textContent).toContain(
      'No likely mistakes in 1 frames',
    );
  });
});
