$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$sdkRoot = Join-Path $projectRoot ".local-tools\android-sdk"
$jdkHome = Join-Path $projectRoot ".local-tools\jdk-21"
$androidDir = Join-Path $PSScriptRoot "android"
$deployDir = Join-Path $projectRoot "deploy"
$apkSource = Join-Path $androidDir "app\build\outputs\apk\debug\app-debug.apk"
$apkTarget = Join-Path $deployDir "Echo-debug.apk"

if (-not (Test-Path (Join-Path $sdkRoot "cmdline-tools\latest\bin\sdkmanager.bat"))) {
  throw "Android SDK not found at $sdkRoot"
}
if (-not (Test-Path (Join-Path $jdkHome "bin\java.exe"))) {
  throw "JDK 21 not found at $jdkHome"
}

$env:ANDROID_HOME = $sdkRoot
$env:ANDROID_SDK_ROOT = $sdkRoot
$env:JAVA_HOME = $jdkHome
$env:Path = "$jdkHome\bin;$sdkRoot\platform-tools;$sdkRoot\cmdline-tools\latest\bin;$env:Path"

$sdkDir = $sdkRoot.Replace("\", "/")
Set-Content -LiteralPath (Join-Path $androidDir "local.properties") -Value "sdk.dir=$sdkDir" -Encoding ASCII

Push-Location $PSScriptRoot
try {
  npm install
  npx cap sync android
} finally {
  Pop-Location
}

Push-Location $androidDir
try {
  .\gradlew.bat assembleDebug
} finally {
  Pop-Location
}

New-Item -ItemType Directory -Force -Path $deployDir | Out-Null
Copy-Item -LiteralPath $apkSource -Destination $apkTarget -Force
Get-Item -LiteralPath $apkTarget
