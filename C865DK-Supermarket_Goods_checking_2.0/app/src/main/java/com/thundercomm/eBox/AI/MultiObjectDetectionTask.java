package com.thundercomm.eBox.AI;

import static java.lang.Math.min;

import android.app.Application;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.Image;
import android.util.Log;

import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Data.Recognition;
import com.thundercomm.eBox.Database.GoodsManager;
import com.thundercomm.eBox.Jni;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.Utils.ImageUtils;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.VIew.MultiObjectDetectionFragment;
import com.thundercomm.eBox.VIew.PlayFragment;
import com.thundercomm.gateway.data.DeviceData;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.Vector;

import lombok.SneakyThrows;

/**
 * Age Gender Detector
 *
 * @Describe
 */
public class MultiObjectDetectionTask {

    private static String TAG = "MultiObjectDetectionTask";

    private HashMap<Integer, Interpreter> mapMultiObjectDetection = new HashMap<Integer, Interpreter>();

    private HashMap<Integer, DataInputFrame> inputFrameMap = new HashMap<Integer, DataInputFrame>();
    private Vector<MultiObjectDetectionTaskThread> mMultiObjectDetectionTaskThreads = new Vector<MultiObjectDetectionTaskThread>();

    private boolean istarting = false;
    private boolean isInit = false;
    private Application mContext;
    private ArrayList<PlayFragment> playFragments;

    private int frameWidth;
    private int frameHeight;

    private static volatile MultiObjectDetectionTask _instance;

    private MultiObjectDetectionTask() {
    }

    public static MultiObjectDetectionTask getMultiObjectDetectionTask() {
        if (_instance == null) {
            synchronized (MultiObjectDetectionTask.class) {
                if (_instance == null) {
                    _instance = new MultiObjectDetectionTask();
                }
            }
        }
        return _instance;
    }

    public void init( Application context, Vector<Integer> idlist, ArrayList<PlayFragment> playFragments, int width, int height) {
        LogUtil.d(TAG, "init AI");
        frameWidth = width;
        frameHeight = height;
        interrupThread();
        for (int i = 0; i < idlist.size(); i++) {
            if (getMultiObjectDetectionAlgorithmType(idlist.elementAt(i))) {
                DataInputFrame data = new DataInputFrame(idlist.elementAt(i));
                inputFrameMap.put(idlist.elementAt(i), data);
            }
        }
        mContext = context;
        istarting = true;
        isInit = true;
        this.playFragments = playFragments;
        for (int i = 0; i < idlist.size(); i++) {
            if (getMultiObjectDetectionAlgorithmType(idlist.elementAt(i))) {
                MultiObjectDetectionTaskThread multiObjectDetectionThreadTaskThread = new MultiObjectDetectionTaskThread(idlist.elementAt(i));
                multiObjectDetectionThreadTaskThread.start();
                mMultiObjectDetectionTaskThreads.add(multiObjectDetectionThreadTaskThread);
            }
        }
    }

    private boolean getMultiObjectDetectionAlgorithmType(int id) {
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(id);
        boolean enable = Boolean.parseBoolean(RtspItemCollection.getInstance().getAttributesValue(deviceData, Constants.ENABLE_MUTILOBJECTDETECTION_STR));
        return enable;
    }

    public void addImgById(int id, final Image img) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.addImgById(img);
    }

    public void addBitmapById(int id, final Bitmap bmp, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addBitMapById(bmp);
    }

    public void addMatById(int id, final Mat img, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addMatById(img);
    }


    class MultiObjectDetectionTaskThread extends Thread {

        private MultiObjectDetectionFragment multiObjectDetectionTask = null;

        private List<String> GoodsList = Arrays.asList(GlobalConfig.mCheckClass);

        private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
        private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";

        // Only return this many results.
        private static final int NUM_DETECTIONS = 10;
        // Number of threads in the java app
        private static final int NUM_THREADS = 4;
        private boolean isModelQuantized = true;
        // Config values.
        private int inputSize = 300;;
        // Pre-allocated buffers.
        private final List<String> labels = new ArrayList<>();
        private int[] intValues;
        // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
        // contains the location of detected boxes
        private float[][][] outputLocations;
        // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
        // contains the classes of detected boxes
        private float[][] outputClasses;
        // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
        // contains the scores of detected boxes
        private float[][] outputScores;
        // numDetections: array of shape [Batchsize]
        // contains the number of detected boxes
        private float[] numDetections;

        private ByteBuffer imgData;

        private Interpreter tfLite;
        int alg_camid = -1;


        private static final int MODEL_INPUT_SIZE = 300;

        //private Detector detector;
        private Bitmap croppedBitmap;
        private Matrix frameToCropTransform;
        private Matrix cropToFrameTransform;

        protected int previewWidth = 0;
        protected int previewHeight = 0;

        MultiObjectDetectionTaskThread(int id) {
            int cropSize = MODEL_INPUT_SIZE;

            croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

            alg_camid = id;

            imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3);
            imgData.order(ByteOrder.nativeOrder());
            intValues = new int[inputSize * inputSize];

            outputLocations = new float[1][NUM_DETECTIONS][4];
            outputClasses = new float[1][NUM_DETECTIONS];
            outputScores = new float[1][NUM_DETECTIONS];
            numDetections = new float[1];

            if (!mapMultiObjectDetection.containsKey(alg_camid)) {
                try {
                    MappedByteBuffer modelFile = loadModelFile(mContext.getAssets(), TF_OD_API_MODEL_FILE);
                    MetadataExtractor metadata = new MetadataExtractor(modelFile);
                    try (BufferedReader br =
                                 new BufferedReader(
                                         new InputStreamReader(
                                                 metadata.getAssociatedFile(TF_OD_API_LABELS_FILE), Charset.defaultCharset()))) {
                        String line;
                        while ((line = br.readLine()) != null) {
                            Log.d(TAG, line);
                            labels.add(line);
                        }
                    }

                    Interpreter.Options options = new Interpreter.Options();
                    options.setNumThreads(NUM_THREADS);
                    options.setUseXNNPACK(true);
                    options.setUseNNAPI(true);
                    tfLite = new Interpreter(modelFile, options);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                mapMultiObjectDetection.put(alg_camid, tfLite);
            } else {
                tfLite = mapMultiObjectDetection.get(alg_camid);
            }
        }

        @SneakyThrows
        @Override
        public void run() {
            super.run();
            Jni.Affinity.bindToCpu(alg_camid % 4 + 4);
            multiObjectDetectionTask = (MultiObjectDetectionFragment) playFragments.get(alg_camid);
            DataInputFrame inputFrame = inputFrameMap.get(alg_camid);
            Mat rotateimage = new Mat(frameHeight, frameWidth, CvType.CV_8UC4);
            Mat frameBgrMat = new Mat(frameHeight, frameWidth, CvType.CV_8UC3);
            LogUtil.d("", "debug test start camid  " + alg_camid);

            while (istarting) {
                try {
                    inputFrame.updateFaceRectCache();
                    Mat mat = inputFrame.getMat();

                    if (!OPencvInit.isLoaderOpenCV() || mat == null) {
                        if (mat != null) mat.release();
                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        continue;
                    }

                    previewWidth = mat.width();
                    previewHeight = mat.height();

                    Core.flip(mat, rotateimage, 0);
                    Imgproc.cvtColor(rotateimage, frameBgrMat, Imgproc.COLOR_BGRA2BGR);
                    Bitmap bitmap = Bitmap.createBitmap(frameBgrMat.width(), frameBgrMat.height(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(frameBgrMat, bitmap);

                    frameToCropTransform =
                            ImageUtils.getTransformationMatrix(mat.width(), mat.height(), frameWidth, frameHeight);
                    cropToFrameTransform = new Matrix();
                    frameToCropTransform.invert(cropToFrameTransform);

                    final Canvas canvas = new Canvas(croppedBitmap);
                    canvas.drawBitmap(bitmap, frameToCropTransform, null);

                    croppedBitmap.getPixels(intValues, 0, croppedBitmap.getWidth(), 0, 0, croppedBitmap.getWidth(), croppedBitmap.getHeight());

                    imgData.rewind();
                    for (int i = 0; i < inputSize; ++i) {
                        for (int j = 0; j < inputSize; ++j) {
                            int pixelValue = intValues[i * inputSize + j];
                            // Quantized model
                            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                            imgData.put((byte) (pixelValue & 0xFF));
                        }
                    }

                    // Copy the input data into TensorFlow.
                    outputLocations = new float[1][NUM_DETECTIONS][4];
                    outputClasses = new float[1][NUM_DETECTIONS];
                    outputScores = new float[1][NUM_DETECTIONS];
                    numDetections = new float[1];

                    Object[] inputArray = {imgData};
                    Map<Integer, Object> outputMap = new HashMap<>();
                    outputMap.put(0, outputLocations);
                    outputMap.put(1, outputClasses);
                    outputMap.put(2, outputScores);
                    outputMap.put(3, numDetections);

                    // Run the inference call.
                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

                    int numDetectionsOutput = min(NUM_DETECTIONS, (int) numDetections[0]);

                    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
                    for (int i = 0; i < numDetectionsOutput; ++i) {
                        final RectF detection =
                                new RectF(outputLocations[0][i][1] * inputSize, outputLocations[0][i][0] * inputSize,
                                        outputLocations[0][i][3] * inputSize, outputLocations[0][i][2] * inputSize);

                        recognitions.add(new Recognition(
                                "" + i, labels.get((int) outputClasses[0][i]), outputScores[0][i], detection, (int) outputClasses[0][i]));
                    }

                    float minimumConfidence = 0.3f;

                    final List<Recognition> mappedRecognitions = new ArrayList<Recognition>();
                    boolean isPerson = false;

                    for (final Recognition result : recognitions) {
                        Log.d(TAG, "recognitions:" + result);
                        if(result.getTitle().equals("person")  && result.getConfidence() >= minimumConfidence) {
                            //mappedRecognitions.clear();
                            //isPerson = true;
                            //break;
                        }
                        final RectF location = result.getLocation();

                        if (location != null && result.getConfidence() >= minimumConfidence && GoodsList.contains(result.getTitle())) {
                            cropToFrameTransform.mapRect(location);

                            result.setLocation(location);
                            mappedRecognitions.add(result);
                        }
                    }
                    if (!isPerson) {
                        GoodsManager.getInstance(mContext).checkResults(mappedRecognitions,
                                multiObjectDetectionTask);
                    }
                    postObjectDetectResult(mappedRecognitions, previewWidth , previewHeight);

                } catch (final Exception e) {
                    e.printStackTrace();
                    LogUtil.e(TAG, "Exception!");
                }
            }
        }

        private void postObjectDetectResult(List<Recognition> results, final int width, final int height) {
            if (multiObjectDetectionTask != null) {
                multiObjectDetectionTask.onDraw(results, width, height);
            }
        }

    }

    public void closeService() {

        isInit = false;
        istarting = false;

        System.gc();
        System.gc();
    }

    private void interrupThread() {
        for (MultiObjectDetectionTaskThread multiObjectDetectionTaskThread : this.mMultiObjectDetectionTaskThreads) {
            if (multiObjectDetectionTaskThread != null && !multiObjectDetectionTaskThread.isInterrupted()) {
                multiObjectDetectionTaskThread.interrupt();
            }
        }
        mapMultiObjectDetection.clear();
    }

    public boolean isIstarting() {
        return isInit;
    }


    private MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
