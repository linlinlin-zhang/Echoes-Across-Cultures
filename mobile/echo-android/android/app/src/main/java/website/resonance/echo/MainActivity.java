package website.resonance.echo;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.webkit.JavascriptInterface;
import android.widget.Toast;
import com.getcapacitor.BridgeActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import org.json.JSONObject;

public class MainActivity extends BridgeActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getBridge() != null && getBridge().getWebView() != null) {
            getBridge().getWebView().addJavascriptInterface(new EchoAndroidBridge(), "EchoAndroid");
        }
    }

    private final class EchoAndroidBridge {
        @JavascriptInterface
        public String savePngBase64(String requestedFilename, String base64Png) {
            try {
                String filename = sanitizeFilename(requestedFilename);
                byte[] bytes = Base64.decode(base64Png, Base64.DEFAULT);
                SaveResult result = savePng(filename, bytes);
                runOnUiThread(() -> Toast.makeText(
                    MainActivity.this,
                    "地图长图已保存到 " + result.displayPath,
                    Toast.LENGTH_LONG
                ).show());
                return new JSONObject()
                    .put("ok", true)
                    .put("uri", result.uri)
                    .put("displayPath", result.displayPath)
                    .toString();
            } catch (Exception error) {
                runOnUiThread(() -> Toast.makeText(
                    MainActivity.this,
                    "地图长图保存失败：" + error.getMessage(),
                    Toast.LENGTH_LONG
                ).show());
                try {
                    return new JSONObject()
                        .put("ok", false)
                        .put("error", error.getMessage())
                        .toString();
                } catch (Exception ignored) {
                    return "{\"ok\":false,\"error\":\"save failed\"}";
                }
            }
        }
    }

    private SaveResult savePng(String filename, byte[] bytes) throws Exception {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = getContentResolver();
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
            values.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
            values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Echo");
            values.put(MediaStore.MediaColumns.IS_PENDING, 1);

            Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                throw new IllegalStateException("无法创建下载文件");
            }

            try (OutputStream stream = resolver.openOutputStream(uri)) {
                if (stream == null) {
                    throw new IllegalStateException("无法写入下载文件");
                }
                stream.write(bytes);
            }

            values.clear();
            values.put(MediaStore.MediaColumns.IS_PENDING, 0);
            resolver.update(uri, values, null, null);
            return new SaveResult(uri.toString(), "下载/Echo/" + filename);
        }

        File directory = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "Echo");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IllegalStateException("无法创建保存目录");
        }
        File outputFile = new File(directory, filename);
        try (OutputStream stream = new FileOutputStream(outputFile)) {
            stream.write(bytes);
        }
        return new SaveResult(Uri.fromFile(outputFile).toString(), outputFile.getAbsolutePath());
    }

    private String sanitizeFilename(String value) {
        String filename = value == null ? "" : value.trim();
        if (filename.isEmpty()) {
            filename = "Echo-Atlas.png";
        }
        filename = filename.replaceAll("[\\\\/:*?\"<>|\\r\\n]+", "-");
        if (!filename.toLowerCase().endsWith(".png")) {
            filename += ".png";
        }
        return filename;
    }

    private static final class SaveResult {
        final String uri;
        final String displayPath;

        SaveResult(String uri, String displayPath) {
            this.uri = uri;
            this.displayPath = displayPath;
        }
    }
}
