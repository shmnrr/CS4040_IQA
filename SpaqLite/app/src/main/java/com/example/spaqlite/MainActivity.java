package com.example.spaqlite;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.IValue;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //setContentView(R.layout.activity_main);

        Module module = null;
        String modelName = "bl_spaq_224x112_cpu_higher_NR";
//        String modelName = "topiq_nr_224x112_cpu_higher_NR";
//        String modelName = "topiq_nr-spaq_224x112_cpu_higher_NR";

        try {
            // Load the lite model
            module = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), (modelName + ".ptl")));
        } catch (IOException e) {
            Log.e("SpaqLite", "Error reading model", e);
            finish();
        }

        // Optional visual feedback
        // Get text view
        //TextView textView = findViewById(R.id.text);
        //StringBuilder results = new StringBuilder();
        // Get image view
        //ImageView imageView = findViewById(R.id.image);

        // Iterate over dataset
        for (int j = 0; j < 1; j++) {
            ArrayList<float[]> results = new ArrayList<>();
            Integer imagesToAsses = 1000;
            long startTimeFull = System.nanoTime();
            for (int i = 1; i < imagesToAsses + 1; i++) {
                long startTime = System.nanoTime();
                Bitmap bitmap = null;
                String imageFile = String.format("%05d.png", i);
                try {
                    bitmap = BitmapFactory.decodeStream(getAssets().open("SPAQ/TestImage/" + imageFile));
                } catch (IOException e) {
                    Log.e("SpaqLite", "Error reading image", e);
                    finish();
                }

                // Show image
                // imageView.setImageBitmap(bitmap);

                // Prepare the ImageLoad object
                ImageLoad prepareImage;
                prepareImage = new ImageLoad(448, 112);

                ArrayList<Bitmap> patches = prepareImage.generatePatches(bitmap);

                // Recycle bitmap
                bitmap.recycle();
                bitmap = null;

                // Initialise the result_sum
                float resultSum = 0;
                // Run the lite model on each patch
                long startTimeInference = System.nanoTime();
                for (Bitmap patch : patches) {

                    Tensor inputTensor;
                    inputTensor = bitmapToFloat32Tensor(patch);

                    // Run the lite model
                    IValue output = module.forward(IValue.from(inputTensor));
                    // Add result for a patch to the result_sum
                    float outputValue = output.toTensor().getDataAsFloatArray()[0];
                    resultSum += outputValue;

                    // Recycle patch bitmap
                    patch.recycle();
                    patch = null;
                }

                long endTime = System.nanoTime();
                float resultMean = resultSum / patches.size();

                float[] resultArray = new float[4];
                resultArray[0] = resultMean;
                resultArray[1] = endTime - startTime;
                resultArray[2] = endTime - startTimeInference;
                resultArray[3] = startTimeInference - startTime;

                results.add(resultArray);
                //textView.setText("Score for image " + i + ":" + result_mean);

            }
            double fullTime = (System.nanoTime() - startTimeFull) / 1000000.0;
//        try {
//            Thread.sleep(2000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//        for (int i = 0; i < results.size(); i++) {
//            float[] resultItem = results.get(i);
//            System.out.println("Result for image " + (i + 1) + " " + resultItem[0] + ", " + resultItem[1] + ", " + resultItem[2] + ", " + resultItem[3]);
//        }

            // Save the results in csv
            try {
                // Set the file name
                //String fileName = "results_" + modelName + "emulated.csv";
                String fileName = "results_" + modelName + "_" + fullTime + ".csv";
                // Create the file object with the desired path
                File file = new File(getFilesDir(), fileName);

                // Open the file for writing
                FileOutputStream fos = new FileOutputStream(file);
                OutputStreamWriter osw = new OutputStreamWriter(fos);

                // Write the header to the file
                String header = "Image name,Score,Total time,Inference time,Preprocessing time\n";
                osw.write(header);

                // Write each result to the file
                for (int i = 0; i < results.size(); i++) {
                    // Format the image name with leading zeros
                    String imageName = String.format("%05d.jpg", i + 1);

                    float[] resultArray = results.get(i);
                    float score = resultArray[0];
                    long time1 = (long) resultArray[1];
                    long time2 = (long) resultArray[2];
                    long time3 = (long) resultArray[3];

                    double time1InMs = time1 / 1000000.0;
                    double time2InMs = time2 / 1000000.0;
                    double time3InMs = time3 / 1000000.0;

                    String output = imageName + "," + score + "," + time1InMs + "," + time2InMs + "," + time3InMs + "\n";
                    osw.write(output);
                }

                // Close the file
                osw.close();
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


    }

        // assetFilePath as in https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

//     Replacement for org.pytorch.torchvision.TensorImageUtils
    public static Tensor bitmapToFloat32Tensor(final Bitmap bitmap,
                                               final float[] meanRgb,
                                               final float[] stdRgb) {
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        final int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        final float[] floatValues = new float[width * height * 3];
        for (int i = 0; i < pixels.length; ++i) {
            final int val = pixels[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - meanRgb[0]) / stdRgb[0];
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - meanRgb[1]) / stdRgb[1];
            floatValues[i * 3 + 2] = ((val & 0xFF) - meanRgb[2]) / stdRgb[2];
        }

        final long[] shape = new long[]{1, 3, height, width};
        return Tensor.fromBlob(floatValues, shape);
    }
    public static Tensor bitmapToFloat32Tensor(final Bitmap bitmap) {
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        final int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        final float[] floatValues = new float[width * height * 3];
        for (int i = 0; i < pixels.length; ++i) {
            final int val = pixels[i];
            floatValues[i * 3 + 0] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }

        final long[] shape = new long[]{1, 3, height, width};
        return Tensor.fromBlob(floatValues, shape);
    }

}