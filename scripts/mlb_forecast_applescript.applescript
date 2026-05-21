-- MLB Forecast launcher.
--
-- Runs Resources/launcher.sh in a fresh login shell. The launcher
-- script handles all the real work (resolving the project root,
-- finding uv, running ``mlb-model app``, surfacing errors).
--
-- We use ``do shell script`` rather than ``open for access`` because
-- ``do shell script`` runs the command through /bin/sh, which AMFI
-- considers a trusted Apple-signed binary -- so the launch never
-- triggers the "unsigned executable" rejection that bites a hand-rolled
-- bundle whose CFBundleExecutable is a bash script.
--
-- The shell command is wrapped in ``set -m`` + ``&`` so the spawned
-- ``uv run mlb-model app`` keeps running after this AppleScript exits.

on run
    set appPath to POSIX path of (path to me as text)
    set launcherPath to appPath & "Contents/Resources/launcher.sh"
    try
        do shell script "set -m; '" & launcherPath & "' >/dev/null 2>&1 &"
    on error errMsg number errNum
        display alert "MLB Forecast" message ("Failed to start the launcher.\n\n" & errMsg & " (#" & errNum & ")\n\nSee ~/Library/Logs/MLBForecast/launcher.log for details.") as critical
    end try
end run
