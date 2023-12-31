package com.example.midaslite;

import androidx.appcompat.app.AppCompatActivity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.graphics.Bitmap;
import android.widget.ImageView;

import com.example.midaslite.databinding.ActivityMainBinding;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'midaslite' library on application startup.
    static {
        System.loadLibrary("midaslite");
        System.loadLibrary("opencv_java4");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        com.example.midaslite.databinding.ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        AssetManager assetManager = this.getAssets();
        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());
        Bitmap imageBitmap = ImageLoader.loadImageFromAsset(this, "img.jpg");
        Mat native_mat = new Mat();
        Utils.bitmapToMat(imageBitmap, native_mat);
        load_asset(native_mat.nativeObj, assetManager);
        if (!native_mat.empty()) {
            // Find the ImageView by its ID
            ImageView imageView = findViewById(R.id.imageView);
            Utils.matToBitmap(native_mat, imageBitmap);
            // Set the loaded Bitmap to the ImageView
            imageView.setImageBitmap(imageBitmap);
        }
    }

    /**
     * A native method that is implemented by the 'midaslite' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public static native void load_asset(long native_mat, AssetManager assetManager);
}