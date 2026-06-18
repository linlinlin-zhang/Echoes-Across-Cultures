# Echo Android

This is a lightweight Capacitor Android shell for the Echo music station.

The APK opens `https://resonance.website/music.html`, while recommendation,
upload, AI lookup, models, and the music catalog stay on the host server.

## Build

This repository uses a local Android SDK and JDK under `.local-tools`.

```powershell
npm install
npx cap sync android
cd android
.\gradlew.bat assembleDebug
```

Or from this directory:

```powershell
.\build-debug.ps1
```

The debug APK is generated at:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

For convenience the script also copies it to:

```text
../../deploy/Echo-debug.apk
```
