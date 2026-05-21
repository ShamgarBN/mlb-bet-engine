"""Scheduled automation jobs (morning sync, weekly retraining).

Each job exposes two entry points:

* a pure-Python function (``run_morning_sync``, ``run_weekly_train``)
  that does the actual work and writes a marker file when it finishes,
* a "should_run_today?" predicate used by the desktop app on launch so
  the user gets fresh data without ever having to install a cron job.

The CLI exposes both as commands ``mlb-model morning-sync`` and
``mlb-model weekly-train`` and an installer ``mlb-model install-schedule``
that drops macOS ``launchd`` plists in ``~/Library/LaunchAgents``.
"""
