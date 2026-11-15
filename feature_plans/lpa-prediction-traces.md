# Prediction traces: choose frames from position, speed and confidence

Covers GitHub issue [#206](https://github.com/paninski-lab/lightning-pose-app/issues/206).

## Where it lives

The Viewer already plays a session with a model's predictions and has **Extract frames → Add**. The traces sit under the videos, above the playback bar, whenever a model is shown (**Prediction traces**, collapsible; the open/closed state is stored as `viewer-view-options.showTraces`).

The traces use the predictions the Viewer has loaded: each shown model × shown view is one source (a select appears when there is more than one), and the keypoints are the ones ticked in the Keypoints panel.

## Signals

| Signal | Computed by | Notes |
|---|---|---|
| Position | Browser, from the loaded CSV | x and y per keypoint |
| Confidence (heatmap) | Browser | Keypoints × time, near-white when confident and orange→red when not, with a vertical colour bar on the right (ticks 0, 0.25, 0.5, 0.75, 0.9, 1); each pixel shows the lowest likelihood it covers |
| Speed, Velocity (x, y), Acceleration | `movement.kinematics` | px/s when the video fps is known, px/frame otherwise |
| Head direction | `movement.kinematics.compute_forward_vector_angle` | Left/right keypoint (guessed from names like `ear_l`), camera view; degrees |
| Distance between keypoints | `movement.kinematics.compute_pairwise_distances` | Two keypoints |

Derived signals are listed by `listPoseSignals` and computed by `computePoseSignal` (`routes/pose_signals.py`). The list is a fixed registry of movement's public API, with typed parameters; no user code runs. The request names the prediction file by its path under the model directory, and a path outside it is refused. **mask < N%** runs `movement.filtering.filter_by_confidence` with the View options likelihood threshold first.

The CSV is read with pandas (x, y, likelihood only) and passed to `movement.io.load_poses.from_numpy`, because EKS outputs carry extra variance columns that `from_lp_file` does not expect.

movement is optional: `pip install "lightning-pose-app[signals]"`. Without it the derived signals are listed but disabled, with the install line; position and confidence still work.

## Stacked traces

**+ Add trace** stacks up to three rows (`trace-row.component.ts`), each with its own model/view, signal, keypoints, axes, parameters, mask and y range. A new row starts as Confidence (or Position after a Confidence row); rows 2–3 have a close button. The rows share one time axis (zooming or panning one zooms all), the time marker, **centre on frame**, the marked segment and the extraction controls, which sit once below the rows. Every row uses the same left and right margins (row labels, colour bar) so the time axes line up. The rows and their choices are saved; settings saved before rows existed become row 1. The head-direction arrow follows the first row that shows head direction. Tested in `prediction-traces.component.spec.ts` "adds up to three trace rows, each with its own signal, and closes them", "zooms every row together and shares the segment", "reads settings saved before there were rows as row 1".

## Choosing what is drawn

A **Keypoints** dropdown (All / None, one checkbox per keypoint) and an **x and y / x / y** choice appear for signals with per-keypoint or per-axis series. The server computes every keypoint and tags each series with its keypoint and axis, so changing the selection never recomputes. A keypoint keeps its colour (tab10) whatever is selected; y is dashed.

## Head-direction arrow

With **Head direction** chosen, each video shows a yellow arrow from the midpoint of the left and right keypoints along movement's forward vector ((right→left) × up, up = −z for `top_down`). It is hidden when either keypoint is below the likelihood threshold or the traces panel is closed. `viewer/prediction-traces/heading-arrow.ts`, tested in `heading-arrow.spec.ts`.

## Plot

Styled after ethograph's pyqtgraph panels: white plot, tick axes (time in seconds when the fps is known), red time marker, blue segment band, legend inside the plot.

`components/time-series-plot/` (canvas, no plotting dependency):

- **Click** jumps the video to that frame.
- **Drag** marks a segment. **Start at current frame** / **End at current frame** do the same from the player.
- **Wheel** zooms time around the pointer, **Ctrl+wheel** or the wheel over the value axis zooms values (the wheel over the time axis zooms time only), **right-drag** zooms each axis separately as in pyqtgraph, **Shift+wheel** or **Alt+drag** (or middle-drag) pans, **double-click** or **A** autoscales. With the plot focused, `+`, `-`, `0` zoom.
- **centre on frame** (ethograph's centre-playback mode): the time marker stays in the middle and the traces slide under it, keeping the zoom span; the window may run past the start or end. Shift+wheel, trackpad swipes and Alt/middle-drag then move through the recording (seek) instead of panning. Remembered with the other choices. Tested in `time-series-plot.component.spec.ts` "keeps the current frame centred in follow mode, and Shift+wheel scrubs" and `prediction-traces.component.spec.ts` "turns on centre-on-frame".
- Head direction: a step across ±180° with a real change of at most 60° (e.g. 178° → −179°) runs to the edge and continues from the other edge. A bigger step is drawn as a jump, because it is usually a tracking error (`crossesWrap`, tested in `time-series-plot.component.spec.ts` "tells a ±180° seam crossing from a real heading jump"). Angles are always drawn frame by frame, never min–max per pixel.
- The **y** min/max fields fix the value range; **auto** goes back to autoscaling. The range is kept per signal and saved in the browser (`prediction-traces.yLimits`). Ctrl+wheel and right-drag on y update the fields. Tested in `prediction-traces.component.spec.ts` "keeps a typed y range per signal until auto is ticked" and `time-series-plot.component.spec.ts` "holds a y range set from outside".
- Lines are drawn as min/max per pixel column, so a one-frame spike stays visible on a long video.

## Extracting from a segment

With a segment marked, **Extract N frames** picks frames from that segment only, with **Diverse (k-means)** (motion energy → PCA → k-means, the default) or **Uniform** (the centre of N equal bins, `options.selection: "uniform"`), and adds the frames to the label file chosen under **Extract frames**. The percentage beside N is the share of the segment's frames.

`extractFrames` with `method: "random"` takes `options.frameRange: [start, end)`. Only those frames are decoded. A segment with fewer frames than requested is a 422 before any work. `frame_selection_kmeans_impl` used to overwrite its `beg_frame` argument with 0; it now honours it.

## Tests

- `app_server/tests/routes/test_pose_signals.py`
  - `test_list_pose_signals_reports_movement_functions`: each choice names its movement function.
  - `test_speed_spikes_where_the_keypoint_jumps`: one value per frame, peak at the injected jump, 10 px/s at 1 px/frame and 10 fps.
  - `test_confidence_threshold_blanks_low_likelihood_frames`: masked frames are `null`.
  - `test_head_direction_needs_two_keypoints`: one series; the same keypoint twice is a 422.
  - `test_prediction_path_outside_model_dir_is_rejected`, `test_unknown_signal_is_rejected`.
  - `test_kmeans_selects_frames_inside_the_segment`: every picked frame is in `[start, end)`.
  - `test_extract_rejects_segment_shorter_than_frame_count`: 422 and nothing written.
- `web_ui/src/app/components/time-series-plot/time-series-plot.component.spec.ts`: a spike survives decimation; click emits the frame; drag marks a segment and does not seek; wheel zoom shows **Reset zoom**, which resets.
- `web_ui/src/app/viewer/prediction-traces/prediction-traces.component.spec.ts`: movement signals are listed; choosing one calls `computePoseSignal` with the prediction path, fps and shown keypoints; head direction guesses `ear_l`/`ear_r`; **Extract from segment** emits the segment and count, and is disabled when the count exceeds the segment; Start/End at current frame set the segment.

## Not done

- Multi-animal predictions: the first individual is used.
- The Labeler's Extract Frames dialog does not take a segment; the Viewer is where the traces are.
