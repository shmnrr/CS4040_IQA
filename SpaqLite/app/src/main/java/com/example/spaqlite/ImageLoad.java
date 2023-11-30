package com.example.spaqlite;

import android.graphics.Bitmap;
import java.util.ArrayList;

public class ImageLoad {
    private int size;
    private int stride;

    public ImageLoad(int size, int stride) {
        this.size = size;
        this.stride = stride;
    }

    public Bitmap adaptiveResize(Bitmap img) {
        int h = img.getHeight();
        int w = img.getWidth();
        if (h < this.size || w < this.size) {
            return img;
        } else {
            return Bitmap.createScaledBitmap(img, this.size, this.size, true);
        }
    }

    public ArrayList<Bitmap> generatePatches(Bitmap image) {
        image = adaptiveResize(image);
        int h = image.getHeight();
        int w = image.getWidth();

        int hIdxMax = h - this.stride;
        int wIdxMax = w - this.stride;

        int hStride = this.stride / 2;
        int wStride = this.stride / 2;

        int hIdxCount = hIdxMax / hStride + (h - this.stride != hIdxMax / hStride * hStride ? 1 : 0) + 1;
        int wIdxCount = wIdxMax / wStride + (w - this.stride != wIdxMax / wStride * wStride ? 1 : 0) + 1;

        ArrayList<Bitmap> patches = new ArrayList<>();

        for (int i = 0; i < hIdxCount; i++) {
            for (int j = 0; j < wIdxCount; j++) {
                int hIdx = i * hStride;
                int wIdx = j * wStride;
                if (hIdx + this.stride <= h && wIdx + this.stride <= w) {
                    patches.add(Bitmap.createBitmap(image, wIdx, hIdx, this.stride, this.stride));
                }
            }
        }

        return patches;
    }
}
