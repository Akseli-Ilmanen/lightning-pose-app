import type { Meta, StoryObj } from '@storybook/angular';
import { TimeSeriesPlotComponent } from './time-series-plot.component';
import { TimeSeries } from './time-series-math';

const N = 3000;

/** A smooth trajectory with a few tracking glitches, like a struggling keypoint. */
function trace(phase: number, glitches: number[]): number[] {
  return Array.from({ length: N }, (_, t) => {
    const glitch = glitches.some((g) => Math.abs(t - g) < 3) ? 80 : 0;
    return 200 + 60 * Math.sin(t / 90 + phase) + 10 * Math.sin(t / 7) + glitch;
  });
}

function confidence(dips: number[]): number[] {
  return Array.from({ length: N }, (_, t) =>
    dips.some((d) => Math.abs(t - d) < 40) ? 0.15 : 0.95,
  );
}

const lines: TimeSeries[] = [
  { label: 'stickTip', values: trace(0, [400, 1710, 2300]) },
  { label: 'beakTip', values: trace(1.5, []) },
];

const heatmap: TimeSeries[] = [
  { label: 'beakTip', values: confidence([]) },
  { label: 'stickTip', values: confidence([400, 1710, 2300]) },
  { label: 'leftEye', values: confidence([1200]) },
  { label: 'rightEye', values: confidence([1250, 2800]) },
];

const meta: Meta<TimeSeriesPlotComponent> = {
  title: 'Components/TimeSeriesPlot',
  component: TimeSeriesPlotComponent,
  args: {
    series: lines,
    nFrames: N,
    currentFrame: 1710,
    mode: 'lines',
    height: 140,
    unit: 'px',
    segment: null,
  },
  render: (args) => ({
    props: args,
    template: `<div style="width: 720px"><app-time-series-plot
      [series]="series" [nFrames]="nFrames" [currentFrame]="currentFrame"
      [mode]="mode" [height]="height" [unit]="unit" [(segment)]="segment"
    ></app-time-series-plot></div>`,
  }),
};

export default meta;
type Story = StoryObj<TimeSeriesPlotComponent>;

export const Position: Story = {};

export const WithSegment: Story = {
  args: { segment: [1500, 2000] },
};

export const ConfidenceHeatmap: Story = {
  args: { series: heatmap, mode: 'heatmap', height: 80, unit: '' },
};

export const MissingValues: Story = {
  args: {
    series: [
      {
        label: 'tail',
        values: trace(0, []).map((v, t) => (t > 1000 && t < 1300 ? null : v)),
      },
    ],
  },
};
