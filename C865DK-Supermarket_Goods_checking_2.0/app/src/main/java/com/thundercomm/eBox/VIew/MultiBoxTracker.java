package com.thundercomm.eBox.VIew;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;

import com.thundercomm.eBox.Utils.ImageUtils;
import com.thundercomm.eBox.Data.Recognition;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class MultiBoxTracker {
    private static final String TAG = "MultiBoxTracker";

    private static final float TEXT_SIZE_DIP = 18;
    private static final float MIN_SIZE = 16.0f;
    private static final int[] COLORS = {
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.WHITE,
            Color.parseColor("#55FF55"),
            Color.parseColor("#FFA500"),
            Color.parseColor("#FF8888"),
            Color.parseColor("#AAAAFF"),
            Color.parseColor("#FFFFAA"),
            Color.parseColor("#55AAAA"),
            Color.parseColor("#AA33AA"),
            Color.parseColor("#0D0068")
    };
    final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
    private final Queue<Integer> availableColors = new LinkedList<Integer>();
    private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
    private final Paint boxPaint = new Paint();
    private final float textSizePx;
    private final BorderedText borderedText;
    private Matrix frameToCanvasMatrix;
    private int frameWidth;
    private int frameHeight;

    public MultiBoxTracker(final Context context) {
        for (final int color : COLORS) {
            availableColors.add(color);
        }

        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Style.STROKE);
        boxPaint.setStrokeWidth(10.0f);
        boxPaint.setStrokeCap(Cap.ROUND);
        boxPaint.setStrokeJoin(Join.ROUND);
        boxPaint.setStrokeMiter(100);

        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
    }

    public synchronized void setFrameConfiguration(
            final int width, final int height) {
        frameWidth = width;
        frameHeight = height;
    }

    public synchronized void trackResults(final List<Recognition> results) {
        processResults(results);
    }

    private Matrix getFrameToCanvasMatrix() {
        return frameToCanvasMatrix;
    }

    public synchronized void draw(final Canvas canvas) {

        final float multiplier =
                Math.min(canvas.getHeight() / (float) (frameHeight), canvas.getWidth() / (float) (frameWidth));
        frameToCanvasMatrix =
                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (frameWidth)),
                        (int) (multiplier * (frameHeight)));
        for (final TrackedRecognition recognition : trackedObjects) {
            final RectF trackedPos = new RectF(recognition.location);

            getFrameToCanvasMatrix().mapRect(trackedPos);
            boxPaint.setColor(recognition.color);

            float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

            final String labelString =
                    !TextUtils.isEmpty(recognition.title)
                            ? String.format("%s", recognition.title)
                            : String.format("%.2f", (100 * recognition.detectionConfidence));
            borderedText.drawText(
                    canvas, trackedPos.left + cornerSize, trackedPos.top, labelString, boxPaint);
        }
    }

    private void processResults(final List<Recognition> results) {
        final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

        screenRects.clear();
        final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

        for (final Recognition result : results) {
            if (result.getLocation() == null) {
                continue;
            }
            final RectF detectionFrameRect = new RectF(result.getLocation());

            final RectF detectionScreenRect = new RectF();
            rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

            Log.d(TAG, "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

            screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

            if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
                Log.d(TAG,"Degenerate rectangle! " + detectionFrameRect);
                continue;
            }

            rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
        }

        trackedObjects.clear();
        if (rectsToTrack.isEmpty()) {
            Log.d(TAG,"Nothing to track, aborting.");
            return;
        }

        for (final Pair<Float, Recognition> potential : rectsToTrack) {
            final TrackedRecognition trackedRecognition = new TrackedRecognition();
            trackedRecognition.detectionConfidence = potential.first;
            trackedRecognition.location = new RectF(potential.second.getLocation());
            trackedRecognition.title = potential.second.getTitle();
            trackedRecognition.color = COLORS[trackedObjects.size()];
            trackedObjects.add(trackedRecognition);

            if (trackedObjects.size() >= COLORS.length) {
                break;
            }
        }
    }

    private static class TrackedRecognition {
        RectF location;
        float detectionConfidence;
        int color;
        String title;
    }
}