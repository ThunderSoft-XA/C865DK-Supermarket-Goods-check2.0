package com.thundercomm.eBox.VIew;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceHolder;
import android.widget.Toast;

import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Data.Recognition;
import com.thundercomm.eBox.Model.RtspItemCollection;

import java.util.HashMap;
import java.util.List;

import androidx.annotation.Nullable;

public class MultiObjectDetectionFragment extends PlayFragment {
    private static final String TAG = "MultiObjectDetectionFragment";
    MultiBoxTracker tracker;
    protected Paint paint_Object;
    HashMap<String, Integer> mCurrentNumHashMap = new HashMap<>();

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        tracker = new MultiBoxTracker(getContext());
        initPaint();
        paint_Object = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint_Object.setColor(Color.CYAN);
        paint_Object.setShadowLayer(10f, 0, 0, Color.CYAN);
        paint_Object.setStyle(Paint.Style.STROKE);
        paint_Object.setStrokeWidth(4);
        paint_Object.setFilterBitmap(true);

        resetCurrentNumHashMap();
    }

    public MultiObjectDetectionFragment(int id) {
        super(id);
    }

    private void draw(final SurfaceHolder mHolder, List<Recognition> results, final int width, final int height) {
        Canvas canvas = null;
        int x = 50;
        int y = 50;
        if (mHolder != null) {
            try {
                canvas = mHolder.lockCanvas();
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                tracker.setFrameConfiguration(width, height);
                tracker.trackResults(results);
                tracker.draw(canvas);

                String msg = "Current Goods Amount:";
                canvas.drawText(msg, x, y, paint_Txt);

                for (final Recognition recognition : results) {
                    mCurrentNumHashMap.put(recognition.getTitle(), mCurrentNumHashMap.get(recognition.getTitle()) + 1);
                }

                for (String key : mCurrentNumHashMap.keySet()) {
                    y += 50;
                    msg = key + ":" + mCurrentNumHashMap.get(key);
                    canvas.drawText(msg, x, y, paint_Txt);
                }
                resetCurrentNumHashMap();
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (null != canvas) {
                    mHolder.unlockCanvasAndPost(canvas);
                }
            }
        }
        hasDrawn = false;
    }

    private void resetCurrentNumHashMap() {
        mCurrentNumHashMap.clear();
        for (int i = 0; i < GlobalConfig.mCheckClass.length; i++) {
            String class_type = GlobalConfig.mCheckClass[i];
            mCurrentNumHashMap.put(class_type, 0);
        }
    }
    public void onDraw(List<Recognition> results,final int width, final int height) {
        draw(mFaceViewHolder, results, width, height);
        hasDrawn = true;
    }

}
